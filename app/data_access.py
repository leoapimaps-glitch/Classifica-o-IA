from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
import csv
import re
import unicodedata

import bcrypt
from openpyxl import load_workbook

from .config import ACCESS_FILE, CLASSIFICATION_FILE, CLIENTS_FILE, RESPONSES_FILE


DAY_NAMES = {
    "seg": 0,
    "segunda": 0,
    "segunda-feira": 0,
    "ter": 1,
    "terca": 1,
    "terça": 1,
    "terca-feira": 1,
    "terça-feira": 1,
    "qua": 2,
    "quarta": 2,
    "quarta-feira": 2,
    "qui": 3,
    "quinta": 3,
    "quinta-feira": 3,
    "sex": 4,
    "sexta": 4,
    "sexta-feira": 4,
    "sab": 5,
    "sábado": 5,
    "sabado": 5,
    "dom": 6,
    "domingo": 6,
}


@dataclass
class AccessDecision:
    allowed: bool
    reason: str = ""


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def digits_only(value: object) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def slugify(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "arquivo"


def workbook_rows(path: Path, sheet_name: str | None = None) -> list[dict[str, object]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[sheet_name] if sheet_name else workbook.worksheets[0]
    iterator = sheet.iter_rows(values_only=True)
    headers_row = next(iterator, ())
    headers = [normalize_text(header).replace(" ", "_") for header in headers_row]
    rows: list[dict[str, object]] = []
    for values in iterator:
        if not any(value not in (None, "") for value in values):
            continue
        row = {
            headers[index] if index < len(headers) else f"coluna_{index + 1}": values[index]
            for index in range(len(values))
            if index < len(headers) and headers[index]
        }
        rows.append(row)
    workbook.close()
    return rows


def load_access_users() -> list[dict[str, object]]:
    if not ACCESS_FILE.exists():
        return []
    return workbook_rows(ACCESS_FILE)


def find_user(login: str) -> dict[str, object] | None:
    normalized_login = normalize_text(login)
    for row in load_access_users():
        if normalize_text(row.get("login")) == normalized_login:
            return row
    return None


def verify_password(raw_password: str, stored_password: object) -> bool:
    stored = str(stored_password or "").strip()
    if not stored:
        return False
    if stored.startswith("$2a$") or stored.startswith("$2b$") or stored.startswith("$2y$"):
        return bcrypt.checkpw(raw_password.encode("utf-8"), stored.encode("utf-8"))
    return raw_password == stored


def parse_date_token(token: str) -> date | None:
    token = token.strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    return None


def parse_time_token(token: str) -> time | None:
    token = token.strip().replace("h", ":")
    for fmt in ("%H:%M", "%H:%M:%S", "%H"):
        try:
            return datetime.strptime(token, fmt).time()
        except ValueError:
            continue
    return None


def evaluate_date_rule(value: object, current_date: date) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if "-" in text and text.count("-") >= 2:
        parsed = parse_date_token(text)
        return parsed == current_date if parsed else False
    separator = None
    for candidate in (" a ", " até ", " to "):
        if candidate in normalize_text(text):
            separator = candidate
            break
    if separator:
        normalized = normalize_text(text)
        left, right = normalized.split(separator, 1)
        start = parse_date_token(left)
        end = parse_date_token(right)
        if start and end:
            return start <= current_date <= end
        return False
    exact = parse_date_token(text)
    return exact == current_date if exact else False


def evaluate_day_rule(value: object, current_date: date) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    chunks = [chunk.strip() for chunk in re.split(r"[,;|]", text) if chunk.strip()]
    weekday = current_date.weekday()
    for chunk in chunks:
        normalized = normalize_text(chunk)
        if normalized in DAY_NAMES and DAY_NAMES[normalized] == weekday:
            return True
        parsed_date = parse_date_token(chunk)
        if parsed_date == current_date:
            return True
    return False


def evaluate_time_rule(value: object, current_time: time) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    ranges = [chunk.strip() for chunk in re.split(r"[,;|]", text) if chunk.strip()]
    for chunk in ranges:
        if "-" not in chunk:
            exact = parse_time_token(chunk)
            if exact and current_time.hour == exact.hour and current_time.minute == exact.minute:
                return True
            continue
        start_raw, end_raw = chunk.split("-", 1)
        start = parse_time_token(start_raw)
        end = parse_time_token(end_raw)
        if not start or not end:
            continue
        if start <= current_time <= end:
            return True
    return False


def evaluate_access_window(user_row: dict[str, object], now: datetime) -> AccessDecision:
    active_value = normalize_text(user_row.get("ativo"))
    if active_value in {"n", "nao", "não", "0", "inativo"}:
        return AccessDecision(False, "Usuario desativado.")

    if user_row.get("data_inicio"):
        start = parse_date_token(str(user_row.get("data_inicio")))
        if start and now.date() < start:
            return AccessDecision(False, "Acesso ainda nao liberado para este usuario.")

    if user_row.get("data_fim"):
        end = parse_date_token(str(user_row.get("data_fim")))
        if end and now.date() > end:
            return AccessDecision(False, "Periodo de acesso encerrado para este usuario.")

    if user_row.get("data") and not evaluate_date_rule(user_row.get("data"), now.date()):
        return AccessDecision(False, "Acesso bloqueado para a data atual.")

    if not evaluate_day_rule(user_row.get("dia"), now.date()):
        return AccessDecision(False, "Acesso indisponivel para o dia atual.")

    if user_row.get("hora_inicio") or user_row.get("hora_fim"):
        start = parse_time_token(str(user_row.get("hora_inicio") or ""))
        end = parse_time_token(str(user_row.get("hora_fim") or ""))
        if start and end and not (start <= now.time() <= end):
            return AccessDecision(False, "Acesso fora da faixa de horario permitida.")

    if not evaluate_time_rule(user_row.get("horario"), now.time()):
        return AccessDecision(False, "Acesso fora da faixa de horario permitida.")

    return AccessDecision(True)


def load_clients() -> list[dict[str, object]]:
    if not CLIENTS_FILE.exists():
        return []
    return workbook_rows(CLIENTS_FILE)


def search_clients(query: str, limit: int = 8) -> list[dict[str, object]]:
    query_text = normalize_text(query)
    query_digits = digits_only(query)
    if not query_text and not query_digits:
        return []

    matches: list[dict[str, object]] = []
    for client in load_clients():
        name = normalize_text(client.get("nome"))
        code = digits_only(client.get("codigo"))
        cnpj = digits_only(client.get("cnpj"))

        if query_digits and (code.startswith(query_digits) or cnpj.startswith(query_digits)):
            matches.append(client)
            continue
        if query_text and query_text in name:
            matches.append(client)

        if len(matches) >= limit:
            break
    return matches


def get_client_by_code(code: str) -> dict[str, object] | None:
    code_digits = digits_only(code)
    if not code_digits:
        return None
    for client in load_clients():
        if digits_only(client.get("codigo")) == code_digits:
            return client
    return None


def load_classification_reference() -> dict[str, list[str]]:
    if not CLASSIFICATION_FILE.exists():
        return {"canais": [], "tipologias": []}
    canal_rows = workbook_rows(CLASSIFICATION_FILE, "canal")
    tipologia_rows = workbook_rows(CLASSIFICATION_FILE, "tipologia")
    return {
        "canais": [str(row.get("canal") or "").strip() for row in canal_rows if str(row.get("canal") or "").strip()],
        "tipologias": [str(row.get("tipologia") or "").strip() for row in tipologia_rows if str(row.get("tipologia") or "").strip()],
    }


def append_submission(record: dict[str, object]) -> None:
    RESPONSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESPONSES_FILE.exists()
    with RESPONSES_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def load_submissions() -> list[dict[str, str]]:
    if not RESPONSES_FILE.exists():
        return []
    with RESPONSES_FILE.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    rows.reverse()
    return rows


def bucket_by_checkout_count(count: int) -> str:
    if count <= 4:
        return "1 a 4"
    if count <= 9:
        return "5 a 9"
    return "10+"
