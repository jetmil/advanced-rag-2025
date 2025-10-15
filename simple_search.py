"""
Простой и быстрый полнотекстовый поиск
Намного лучше векторных баз для справочной информации
"""
import re
import sys
from pathlib import Path

def search_text(query: str, text_file: str, context_lines: int = 5):
    """
    Поиск с контекстом (как grep -C)

    Args:
        query: поисковый запрос (поддерживает regex)
        text_file: путь к файлу
        context_lines: сколько строк до и после показать
    """
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    pattern = re.compile(query, re.IGNORECASE)

    for i, line in enumerate(lines):
        if pattern.search(line):
            # Берем контекст
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)

            context = ''.join(lines[start:end])
            results.append({
                'line_num': i + 1,
                'context': context
            })

    return results

def main():
    project_dir = Path(__file__).parent
    text_file = str(project_dir / "cosmic_texts.txt")

    print("="*70)
    print("БЫСТРЫЙ ПОЛНОТЕКСТОВЫЙ ПОИСК")
    print("="*70)
    print("Команды:")
    print("  /quit - выход")
    print("  /context N - установить контекст (по умолчанию 5 строк)")
    print("="*70)

    context_lines = 5

    while True:
        query = input("\n🔍 Поиск: ").strip()

        if not query:
            continue

        if query.lower() in ['/quit', '/exit', '/q']:
            print("Пока!")
            break

        if query.startswith('/context '):
            try:
                context_lines = int(query.split()[1])
                print(f"✅ Контекст установлен: {context_lines} строк")
            except:
                print("❌ Используйте: /context 5")
            continue

        results = search_text(query, text_file, context_lines)

        print(f"\n{'='*70}")
        print(f"Найдено: {len(results)} совпадений")
        print(f"{'='*70}")

        if not results:
            print("❌ Ничего не найдено")
            continue

        for i, result in enumerate(results[:10], 1):  # Показываем первые 10
            print(f"\n[{i}] Строка {result['line_num']}:")
            print("-"*70)
            print(result['context'])
            print("-"*70)

        if len(results) > 10:
            print(f"\n... и еще {len(results) - 10} совпадений")

if __name__ == "__main__":
    main()
