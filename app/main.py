from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import shutil
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from openpyxl import Workbook
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .config import ADMIN_LOGIN, ADMIN_PASSWORD, SESSION_SECRET, STATIC_DIR, TEMPLATES_DIR, TIMEZONE, UPLOADS_DIR
from .data_access import (
    append_submission,
    evaluate_access_window,
    find_user,
    get_client_by_code,
    load_access_users,
    load_classification_reference,
    load_submissions,
    normalize_text,
    search_clients,
    slugify,
    verify_password,
)
from .vision import estimate_checkouts_from_image
from .google_sync import sync_submission


app = FastAPI(title="Classificacao de Clientes")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def now_sp() -> datetime:
    return datetime.now(TIMEZONE)


def login_redirect(message: str = "") -> RedirectResponse:
    if not message:
        return RedirectResponse("/login", status_code=303)
    return RedirectResponse(f"/login?error={quote(message)}", status_code=303)


def current_user(request: Request) -> tuple[dict[str, object] | None, str]:
    login = request.session.get("login")
    if not login:
        return None, ""
    user = find_user(str(login))
    if not user:
        request.session.clear()
        return None, "Usuario nao encontrado na planilha de acessos."
    decision = evaluate_access_window(user, now_sp())
    if not decision.allowed:
        request.session.clear()
        return None, decision.reason
    return user, ""


def admin_authenticated(request: Request) -> bool:
    return bool(request.session.get("admin_login"))


def admin_login_allowed(login: str, password: str) -> bool:
    if login == ADMIN_LOGIN and password == ADMIN_PASSWORD:
        return True

    # Suporte opcional para admin vindo da planilha acessos.xlsx com coluna Perfil=admin.
    user = find_user(login)
    if not user or not verify_password(password, user.get("senha")):
        return False
    return normalize_text(user.get("perfil")) == "admin"


@app.get("/")
def root(request: Request):
    user, _ = current_user(request)
    target = "/formulario" if user else "/login"
    return RedirectResponse(target, status_code=303)


@app.get("/login")
def login_page(request: Request):
    user, _ = current_user(request)
    if user:
        return RedirectResponse("/formulario", status_code=303)
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "error": request.query_params.get("error", ""),
            "has_users": bool(load_access_users()),
        },
    )


@app.post("/login")
async def login(request: Request, login: str = Form(...), senha: str = Form(...)):
    user = find_user(login)
    if not user or not verify_password(senha, user.get("senha")):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Login ou senha invalidos.", "has_users": bool(load_access_users())},
            status_code=401,
        )

    decision = evaluate_access_window(user, now_sp())
    if not decision.allowed:
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": decision.reason, "has_users": True},
            status_code=403,
        )

    request.session["login"] = str(user.get("login") or login)
    return RedirectResponse("/formulario", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/admin/login")
def admin_login_page(request: Request):
    if admin_authenticated(request):
        return RedirectResponse("/admin", status_code=303)
    return templates.TemplateResponse(
        request,
        "admin_login.html",
        {
            "error": request.query_params.get("error", ""),
        },
    )


@app.post("/admin/login")
async def admin_login(request: Request, login: str = Form(...), senha: str = Form(...)):
    if not admin_login_allowed(login, senha):
        return templates.TemplateResponse(
            request,
            "admin_login.html",
            {"error": "Credenciais de admin invalidas."},
            status_code=401,
        )

    request.session["admin_login"] = login
    return RedirectResponse("/admin", status_code=303)


@app.get("/admin/logout")
def admin_logout(request: Request):
    request.session.pop("admin_login", None)
    return RedirectResponse("/admin/login", status_code=303)


@app.get("/admin")
def admin_dashboard(request: Request):
    if not admin_authenticated(request):
        return RedirectResponse("/admin/login", status_code=303)

    submissions = load_submissions()
    return templates.TemplateResponse(
        request,
        "admin_dashboard.html",
        {
            "admin_login": request.session.get("admin_login", "admin"),
            "submissions": submissions,
            "total": len(submissions),
        },
    )


def _submission_headers() -> list[str]:
    rows = load_submissions()
    if not rows:
        return [
            "data_hora",
            "usuario_login",
            "nome_vendedor",
            "codigo_cliente",
            "cnpj",
            "nome_loja",
            "cidade",
            "uf",
            "canal",
            "foto_fachada",
            "foto_interna",
            "qtd_checkouts",
            "classificacao_checkout",
            "status_classificacao",
        ]
    return list(rows[0].keys())


@app.get("/admin/export.csv")
def admin_export_csv(request: Request):
    if not admin_authenticated(request):
        return RedirectResponse("/admin/login", status_code=303)

    rows = load_submissions()
    headers = _submission_headers()
    lines = [",".join(headers)]
    for row in rows:
        values = [str(row.get(col, "")).replace('"', '""') for col in headers]
        lines.append(",".join(f'"{value}"' for value in values))
    payload = "\n".join(lines).encode("utf-8")
    filename = f"respostas_{now_sp().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=payload,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/admin/export.xlsx")
def admin_export_xlsx(request: Request):
    if not admin_authenticated(request):
        return RedirectResponse("/admin/login", status_code=303)

    rows = load_submissions()
    headers = _submission_headers()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Respostas"
    sheet.append(headers)
    for row in rows:
        sheet.append([str(row.get(col, "")) for col in headers])

    stream = BytesIO()
    workbook.save(stream)
    stream.seek(0)
    filename = f"respostas_{now_sp().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return Response(
        content=stream.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/formulario")
def formulario(request: Request):
    user, error = current_user(request)
    if not user:
        return login_redirect(error)

    reference = load_classification_reference()
    return templates.TemplateResponse(
        request,
        "formulario.html",
        {
            "user": user,
            "reference": reference,
            "submitted": request.query_params.get("submitted", "0"),
        },
    )


@app.get("/api/clientes")
def api_clientes(request: Request, q: str = ""):
    user, error = current_user(request)
    if not user:
        return JSONResponse({"detail": error or "Nao autenticado."}, status_code=401)

    clients = search_clients(q)
    payload = [
        {
            "codigo": str(client.get("codigo") or ""),
            "cnpj": str(client.get("cnpj") or ""),
            "nome": str(client.get("nome") or ""),
            "cidade": str(client.get("cidade") or ""),
            "uf": str(client.get("uf") or ""),
            "canal": str(client.get("canal") or ""),
            "endereco": str(client.get("endereco") or ""),
        }
        for client in clients
    ]
    return JSONResponse(payload)


def save_upload(file: UploadFile, login: str, client_code: str, label: str) -> str:
    date_folder = now_sp().strftime("%Y-%m-%d")
    destination_dir = UPLOADS_DIR / date_folder / slugify(login) / slugify(client_code)
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or label).suffix or ".jpg"
    filename = f"{label}-{now_sp().strftime('%H%M%S')}{suffix}"
    destination_path = destination_dir / filename
    with destination_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return str(destination_path.relative_to(UPLOADS_DIR)).replace("\\", "/")


@app.post("/visitas")
async def criar_visita(
    request: Request,
    nome_vendedor: str = Form(...),
    codigo_cliente: str = Form(...),
    foto_fachada: UploadFile = File(...),
    foto_interna: UploadFile = File(...),
):
    user, error = current_user(request)
    if not user:
        return login_redirect(error)

    client = get_client_by_code(codigo_cliente)
    if not client:
        return RedirectResponse("/formulario?submitted=0", status_code=303)

    fachada_path = save_upload(foto_fachada, str(user.get("login") or "usuario"), codigo_cliente, "fachada")
    interna_path = save_upload(foto_interna, str(user.get("login") or "usuario"), codigo_cliente, "interna")
    interna_full_path = UPLOADS_DIR / interna_path

    checkout_count, checkout_bucket, vision_status = estimate_checkouts_from_image(interna_full_path)

    created_at = now_sp()
    submission_record = {
        "data_hora": created_at.isoformat(timespec="seconds"),
        "usuario_login": str(user.get("login") or ""),
        "nome_vendedor": nome_vendedor.strip(),
        "codigo_cliente": str(client.get("codigo") or ""),
        "cnpj": str(client.get("cnpj") or ""),
        "nome_loja": str(client.get("nome") or ""),
        "cidade": str(client.get("cidade") or ""),
        "uf": str(client.get("uf") or ""),
        "canal": str(client.get("canal") or ""),
        "foto_fachada": fachada_path,
        "foto_interna": interna_path,
        "qtd_checkouts": checkout_count,
        "classificacao_checkout": checkout_bucket,
        "status_classificacao": vision_status,
    }

    cloud_sync = sync_submission(
        created_at=created_at,
        login=str(user.get("login") or "usuario"),
        codigo_cliente=str(client.get("codigo") or "sem-codigo"),
        fachada_local_path=UPLOADS_DIR / fachada_path,
        interna_local_path=UPLOADS_DIR / interna_path,
        row_values=list(submission_record.values()),
    )

    if cloud_sync.synced:
        submission_record["foto_fachada"] = cloud_sync.fachada_url
        submission_record["foto_interna"] = cloud_sync.interna_url
        submission_record["status_classificacao"] = (
            "ok_google" if vision_status == "ok" else f"fallback_google_{vision_status}"
        )

    append_submission(submission_record)

    request.session["last_submission"] = {
        "cliente": str(client.get("nome") or ""),
        "codigo": str(client.get("codigo") or ""),
        "canal": str(client.get("canal") or ""),
        "qtd_checkouts": checkout_count,
        "classificacao_checkout": checkout_bucket,
        "status": "Analise por imagem finalizada" if vision_status == "ok" else "Analise por imagem com fallback",
        "cloud_status": cloud_sync.message,
    }
    return RedirectResponse("/resultado", status_code=303)


@app.get("/resultado")
def resultado(request: Request):
    user, error = current_user(request)
    if not user:
        return login_redirect(error)
    submission = request.session.get("last_submission")
    if not submission:
        return RedirectResponse("/formulario", status_code=303)
    return templates.TemplateResponse(request, "resultado.html", {"submission": submission, "user": user})


@app.get("/health")
def healthcheck():
    return {"status": "ok"}
