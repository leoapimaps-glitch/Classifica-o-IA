from getpass import getpass

import bcrypt


def main() -> None:
    password = getpass("Senha para gerar hash: ")
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    print(hashed)


if __name__ == "__main__":
    main()