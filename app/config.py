from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import os


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"

ACCESS_FILE = BASE_DIR / "acessos.xlsx"
CLIENTS_FILE = BASE_DIR / "Base_clientes.xlsx"
CLASSIFICATION_FILE = BASE_DIR / "Classificação.xlsx"
RESPONSES_FILE = DATA_DIR / "respostas.csv"

SESSION_SECRET = os.getenv("SESSION_SECRET", "troque-esta-chave-em-producao")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_URL = os.getenv(
	"HF_MODEL_URL",
	"https://api-inference.huggingface.co/models/google/owlvit-base-patch32",
)
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
GOOGLE_SHEETS_SPREADSHEET_ID = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "")
GOOGLE_SHEETS_TAB = os.getenv("GOOGLE_SHEETS_TAB", "Respostas")
ADMIN_LOGIN = os.getenv("ADMIN_LOGIN", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(BASE_DIR / "models" / "checkout_yolo.pt"))
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.30"))

try:
	TIMEZONE = ZoneInfo("America/Sao_Paulo")
except ZoneInfoNotFoundError:
	TIMEZONE = ZoneInfo("UTC")
