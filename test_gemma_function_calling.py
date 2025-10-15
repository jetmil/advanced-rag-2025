"""
Тест поддержки Function Calling в Gemma 3-27B
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Простой тест function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Получить погоду в городе",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

messages = [
    {"role": "system", "content": "Ты - помощник с доступом к инструментам."},
    {"role": "user", "content": "Какая погода в Москве?"}
]

print("=" * 70)
print("ТЕСТ FUNCTION CALLING: Gemma 3-27B")
print("=" * 70)

try:
    response = client.chat.completions.create(
        model="google/gemma-3-27b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.3,
        max_tokens=500
    )

    message = response.choices[0].message

    print("\n[OK] Zapros vypolnen uspeshno!")
    print(f"\nSoderzhanie otveta: {message.content}")
    print(f"\nEst tool_calls: {hasattr(message, 'tool_calls') and message.tool_calls is not None}")

    if hasattr(message, 'tool_calls') and message.tool_calls:
        print(f"\n[SUCCESS] FUNCTION CALLING PODDERZHIVAETSYA!")
        for tc in message.tool_calls:
            print(f"   - Funktsiya: {tc.function.name}")
            print(f"   - Argumenty: {tc.function.arguments}")
    else:
        print(f"\n[FAIL] FUNCTION CALLING NE PODDERZHIVAETSYA")
        print("   Model prosto otvetila tekstom vmesto vyzova funktsii")

except Exception as e:
    print(f"\n[ERROR] OSHIBKA: {e}")
    print("\nВозможные причины:")
    print("1. Gemma 3-27B не поддерживает OpenAI function calling format")
    print("2. LM Studio не передаёт tools в модель")
    print("3. Нужна специальная версия с суффиксом -function-calling")

print("\n" + "=" * 70)
