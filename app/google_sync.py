from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import json

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from .config import (
    GOOGLE_DRIVE_FOLDER_ID,
    GOOGLE_SERVICE_ACCOUNT_FILE,
    GOOGLE_SERVICE_ACCOUNT_JSON,
    GOOGLE_SHEETS_SPREADSHEET_ID,
    GOOGLE_SHEETS_TAB,
)


SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

_folder_cache: dict[tuple[str, str], str] = {}


@dataclass
class SyncResult:
    enabled: bool
    synced: bool
    fachada_url: str = ""
    interna_url: str = ""
    message: str = ""


def _escape_query(value: str) -> str:
    return value.replace("'", "\\'")


def _credentials_from_env() -> Credentials | None:
    if GOOGLE_SERVICE_ACCOUNT_JSON.strip():
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        return Credentials.from_service_account_info(info, scopes=SCOPES)
    if GOOGLE_SERVICE_ACCOUNT_FILE.strip():
        return Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return None


@lru_cache(maxsize=1)
def _drive_service():
    creds = _credentials_from_env()
    if not creds:
        return None
    return build("drive", "v3", credentials=creds, cache_discovery=False)


@lru_cache(maxsize=1)
def _sheets_service():
    creds = _credentials_from_env()
    if not creds:
        return None
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def google_sync_enabled() -> bool:
    return bool(_credentials_from_env() and GOOGLE_DRIVE_FOLDER_ID and GOOGLE_SHEETS_SPREADSHEET_ID)


def _find_or_create_folder(name: str, parent_id: str) -> str:
    cached = _folder_cache.get((parent_id, name))
    if cached:
        return cached

    drive = _drive_service()
    if not drive:
        raise RuntimeError("Drive indisponivel por falta de credencial.")

    query = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and name='{_escape_query(name)}' "
        f"and '{parent_id}' in parents and trashed=false"
    )
    response = drive.files().list(q=query, fields="files(id,name)", pageSize=1).execute()
    files = response.get("files", [])
    if files:
        folder_id = files[0]["id"]
        _folder_cache[(parent_id, name)] = folder_id
        return folder_id

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    created = drive.files().create(body=metadata, fields="id").execute()
    folder_id = created["id"]
    _folder_cache[(parent_id, name)] = folder_id
    return folder_id


def _upload_file(local_path: Path, parent_id: str, remote_name: str) -> str:
    drive = _drive_service()
    if not drive:
        raise RuntimeError("Drive indisponivel por falta de credencial.")

    metadata = {"name": remote_name, "parents": [parent_id]}
    media = MediaIoBaseUpload(BytesIO(local_path.read_bytes()), mimetype="image/jpeg", resumable=False)
    created = drive.files().create(body=metadata, media_body=media, fields="id,webViewLink").execute()
    return str(created.get("webViewLink") or f"https://drive.google.com/file/d/{created['id']}/view")


def _append_sheet_row(values: list[object]) -> None:
    sheets = _sheets_service()
    if not sheets:
        raise RuntimeError("Sheets indisponivel por falta de credencial.")
    sheets.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEETS_SPREADSHEET_ID,
        range=f"{GOOGLE_SHEETS_TAB}!A1",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": [values]},
    ).execute()


def sync_submission(
    created_at: datetime,
    login: str,
    codigo_cliente: str,
    fachada_local_path: Path,
    interna_local_path: Path,
    row_values: list[object],
) -> SyncResult:
    if not google_sync_enabled():
        return SyncResult(enabled=False, synced=False, message="Integracao Google nao configurada.")

    try:
        date_folder = created_at.strftime("%Y-%m-%d")
        day_folder = _find_or_create_folder(date_folder, GOOGLE_DRIVE_FOLDER_ID)
        login_folder = _find_or_create_folder(login, day_folder)
        client_folder = _find_or_create_folder(codigo_cliente, login_folder)

        fachada_url = _upload_file(fachada_local_path, client_folder, fachada_local_path.name)
        interna_url = _upload_file(interna_local_path, client_folder, interna_local_path.name)

        cloud_row = list(row_values)
        if len(cloud_row) >= 11:
            cloud_row[9] = fachada_url
            cloud_row[10] = interna_url
        _append_sheet_row(cloud_row)
        return SyncResult(enabled=True, synced=True, fachada_url=fachada_url, interna_url=interna_url, message="Sincronizado no Google.")
    except Exception as exc:
        return SyncResult(enabled=True, synced=False, message=f"Falha de sincronizacao Google: {exc}")
