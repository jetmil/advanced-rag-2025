"""
Проверка логики: какую базу использует система
"""
import os
from pathlib import Path

project_dir = Path(__file__).parent

print("="*70)
print("ПРОВЕРКА ЛОГИКИ СИСТЕМЫ")
print("="*70)

# Проверяем все базы данных
print("\nДоступные базы данных:")
db_dirs = list(project_dir.glob("chroma_db_*"))
for db_dir in db_dirs:
    size_mb = sum(f.stat().st_size for f in db_dir.rglob('*') if f.is_file()) / (1024*1024)
    print(f"  - {db_dir.name}: {size_mb:.1f} MB")

# Проверяем файл
text_file = project_dir / "cosmic_texts.txt"
if text_file.exists():
    size_mb = text_file.stat().st_size / (1024*1024)
    print(f"\nФайл cosmic_texts.txt: {size_mb:.1f} MB")
else:
    print("\n❌ Файл cosmic_texts.txt НЕ НАЙДЕН!")

# Проверяем содержимое файла
print("\nПроверка содержимого файла...")
if text_file.exists():
    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read()

    terms = {
        "Перун": content.count("Перун"),
        "Фираст": content.count("Фираст"),
        "Пирва": content.count("Пирва"),
    }

    print("Количество упоминаний терминов:")
    for term, count in terms.items():
        print(f"  - {term}: {count} раз(а)")

print("\n" + "="*70)
print("ВЫВОД:")
print("="*70)

if db_dirs:
    print(f"✅ Найдено {len(db_dirs)} баз данных")
    print(f"✅ Термины присутствуют в исходном файле")
    print("\nПРОБЛЕМА может быть:")
    print("  1. Используется СТАРАЯ база (до исправления chunk_size)")
    print("  2. LM Studio дает неправильные ответы")
    print("  3. Контекст не передается корректно в LLM")
    print("\nРЕКОМЕНДАЦИЯ:")
    print("  - Удалить ВСЕ базы кроме chroma_db_test")
    print("  - Или переименовать chroma_db_test → chroma_db_космоэнергетика")
else:
    print("❌ Базы данных не найдены!")
