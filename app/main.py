from __future__ import annotations

import csv
from datetime import datetime
from io import BytesIO
import json
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
    bucket_by_checkout_count,
    evaluate_access_window,
    find_user,
    get_submission_by_row_id,
    get_client_by_code,
    load_access_users,
    load_classification_reference,
    load_submissions,
    normalize_text,
    search_clients,
    slugify,
    update_submission,
    verify_password,
)
from .vision import estimate_checkouts_from_image
from .google_sync import sync_submission


UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR = Path(__file__).resolve().parent.parent / "treino IA"
TRAINING_IMAGES_DIR = TRAINING_DIR / "images"
TRAINING_LABELS_DIR = TRAINING_DIR / "labels"
TRAINING_META_FILE = TRAINING_DIR / "annotations.csv"
TRAINING_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_LABELS_DIR.mkdir(parents=True, exist_ok=True)


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
            "feedback": request.query_params.get("feedback", ""),
        },
    )


def _resolve_submission_image(value: str) -> tuple[str, Path | None]:
    text = str(value or "").strip()
    if not text:
        return "", None

    if text.startswith("http://") or text.startswith("https://"):
        return text, None

    local_path = UPLOADS_DIR / text
    if local_path.exists():
        return f"/uploads/{text}", local_path
    return text, None


def _persist_training_annotation(row_id: str, local_image_path: Path | None, boxes: list[dict[str, float]], true_count: int) -> None:
    timestamp = now_sp().strftime("%Y%m%d_%H%M%S")
    base_name = f"checkout_{timestamp}_row{row_id}"

    image_file_name = ""
    if local_image_path and local_image_path.exists():
        image_file_name = f"{base_name}{local_image_path.suffix.lower() or '.jpg'}"
        target_image = TRAINING_IMAGES_DIR / image_file_name
        shutil.copy2(local_image_path, target_image)

    label_file = TRAINING_LABELS_DIR / f"{base_name}.txt"
    with label_file.open("w", encoding="utf-8") as handle:
        for box in boxes:
            x = max(0.0, min(float(box.get("x", 0.0)), 1.0))
            y = max(0.0, min(float(box.get("y", 0.0)), 1.0))
            w = max(0.0, min(float(box.get("w", 0.0)), 1.0))
            h = max(0.0, min(float(box.get("h", 0.0)), 1.0))
            cx = x + (w / 2)
            cy = y + (h / 2)
            handle.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    file_exists = TRAINING_META_FILE.exists()
    with TRAINING_META_FILE.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["created_at", "row_id", "image_file", "label_file", "true_count", "boxes"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "created_at": now_sp().isoformat(timespec="seconds"),
                "row_id": row_id,
                "image_file": image_file_name,
                "label_file": label_file.name,
                "true_count": true_count,
                "boxes": len(boxes),
            }
        )


@app.get("/admin/ajustar/{row_id}")
def admin_adjust_page(request: Request, row_id: str):
    if not admin_authenticated(request):
        return RedirectResponse("/admin/login", status_code=303)

    row = get_submission_by_row_id(row_id)
    if not row:
        return RedirectResponse("/admin?feedback=Envio+nao+encontrado", status_code=303)

    image_src, local_image_path = _resolve_submission_image(row.get("foto_interna", ""))
    return templates.TemplateResponse(
        request,
        "admin_annotate.html",
        {
            "admin_login": request.session.get("admin_login", "admin"),
            "row": row,
            "image_src": image_src,
            "has_local_image": bool(local_image_path),
            "feedback": request.query_params.get("feedback", ""),
        },
    )


@app.post("/admin/ajustar/{row_id}")
async def admin_adjust_submit(
    request: Request,
    row_id: str,
    true_count: int = Form(...),
    boxes_json: str = Form("[]"),
):
    if not admin_authenticated(request):
        return RedirectResponse("/admin/login", status_code=303)

    row = get_submission_by_row_id(row_id)
    if not row:
        return RedirectResponse("/admin?feedback=Envio+nao+encontrado", status_code=303)

    try:
        parsed_boxes = json.loads(boxes_json or "[]")
        if not isinstance(parsed_boxes, list):
            parsed_boxes = []
    except json.JSONDecodeError:
        parsed_boxes = []

    boxes: list[dict[str, float]] = []
    for item in parsed_boxes:
        if not isinstance(item, dict):
            continue
        try:
            box = {
                "x": float(item.get("x", 0.0)),
                "y": float(item.get("y", 0.0)),
                "w": float(item.get("w", 0.0)),
                "h": float(item.get("h", 0.0)),
            }
        except (TypeError, ValueError):
            continue
        if box["w"] <= 0 or box["h"] <= 0:
            continue
        boxes.append(box)

    corrected_count = max(0, min(int(true_count), 60))
    corrected_bucket = bucket_by_checkout_count(corrected_count)

    updated = update_submission(
        row_id,
        {
            "qtd_checkouts_ajustado": str(corrected_count),
            "classificacao_checkout_ajustada": corrected_bucket,
            "status_classificacao": f"ajuste_manual_{len(boxes)}_boxes",
        },
    )
    if not updated:
        return RedirectResponse("/admin?feedback=Falha+ao+atualizar+registro", status_code=303)

    _, local_image_path = _resolve_submission_image(row.get("foto_interna", ""))
    _persist_training_annotation(row_id=row_id, local_image_path=local_image_path, boxes=boxes, true_count=corrected_count)

    return RedirectResponse("/admin?feedback=Ajuste+salvo+e+enviado+para+treino", status_code=303)


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
            "segmento",
            "foto_fachada",
            "foto_interna",
            "qtd_checkouts",
            "qtd_checkouts_ajustado",
            "classificacao_checkout",
            "classificacao_checkout_ajustada",
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
    segmento: str = Form(...),
    foto_fachada: UploadFile = File(...),
    foto_interna: UploadFile = File(...),
):
    user, error = current_user(request)
    if not user:
        return login_redirect(error)

    client = get_client_by_code(codigo_cliente)
    if not client:
        return RedirectResponse("/formulario?submitted=0", status_code=303)

    reference = load_classification_reference()
    segmentos = [str(value).strip() for value in reference.get("segmentos", []) if str(value).strip()]
    segmento_normalized = normalize_text(segmento)
    segmentos_normalizados = {normalize_text(value) for value in segmentos}
    if segmentos and segmento_normalized not in segmentos_normalizados:
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
        "segmento": segmento.strip(),
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
            "ok_google" if vision_status.startswith("ok") else f"fallback_google_{vision_status}"
        )

    append_submission(submission_record)

    request.session["last_submission"] = {
        "cliente": str(client.get("nome") or ""),
        "codigo": str(client.get("codigo") or ""),
        "canal": str(client.get("canal") or ""),
        "qtd_checkouts": checkout_count,
        "classificacao_checkout": checkout_bucket,
        "status": (
            "Analise por imagem finalizada"
            if vision_status.startswith("ok")
            else f"Analise por imagem com fallback ({vision_status})"
        ),
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
