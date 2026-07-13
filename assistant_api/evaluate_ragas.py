"""
Оценка качества RAG системы через RAGAS для assistant_api.
Использует OpenAI API для RAG и для метрик RAGAS.
"""

import math
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._faithfulness import Faithfulness

from rag_pipeline import RAGPipeline


# Тестовые вопросы и эталонные ответы для rag_architecture.md
EVALUATION_DATA = [
    {
        "question": "Почему vector-only RAG недостаточен для описанной системы?",
        "ground_truth": (
            "Чистый vector-only RAG недостаточен, потому что нужен гибридный retrieval: "
            "семантический поиск по прозе плюс exact-match/graph поиск по кодам, шагам и состояниям."
        ),
    },
    {
        "question": "Какие компоненты входят в ingestion pipeline?",
        "ground_truth": (
            "Ingestion pipeline включает parse, normalize, chunk и enrich: парсинг источников, "
            "нормализацию документов, чанкование и обогащение метаданными."
        ),
    },
    {
        "question": "Что такое hybrid retrieval в этой архитектуре?",
        "ground_truth": (
            "Hybrid retrieval параллельно использует exact lookup по кодам, sparse BM25-поиск "
            "и dense vector similarity, после чего результаты объединяются через RRF."
        ),
    },
    {
        "question": "Какие три слоя хранилища знаний предусмотрены?",
        "ground_truth": (
            "Три слоя: векторное хранилище для semantic layer, структурное PostgreSQL-хранилище "
            "и граф ссылок для reference graph."
        ),
    },
    {
        "question": "Какие эпистемические статусы должна различать генерация?",
        "ground_truth": (
            "Генерация должна различать три статуса: описано в сценарии, явно исключено "
            "и открытый вопрос или не покрытый edge case."
        ),
    },
]

EVALUATION_QUESTIONS = [item["question"] for item in EVALUATION_DATA]
RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def prepare_dataset(pipeline: RAGPipeline, evaluation_data: list) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов и эталонных ответов.

    Args:
        pipeline: RAG pipeline для получения ответов
        evaluation_data: список словарей question/ground_truth

    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []

    print("[*] Получение ответов от RAG системы...\n")

    for i, item in enumerate(evaluation_data, 1):
        question = item["question"]
        print(f"  {i}/{len(evaluation_data)}: {question}")

        result = pipeline.query(question, use_cache=False)

        questions_list.append(question)
        answers_list.append(result["answer"])
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        ground_truths_list.append(item["ground_truth"])

        print("     [+] Ответ получен от OpenAI API")

    print()

    return Dataset.from_dict({
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list,
    })


def _safe_mean(values: list) -> float:
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else 0.0


def evaluate_rag_system():
    """Основная функция оценки RAG-системы через RAGAS."""
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (API MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()

    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        sys.exit(1)

    try:
        print("[*] Инициализация RAG системы (API mode)...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_eval_collection",
            cache_db_path="api_rag_eval_cache.db",
            data_file="data/rag_architecture.md",
            persist_directory="./chroma_eval_db",
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)

    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_DATA)
    print("=" * 70)

    print("\n[*] Запуск оценки метрик RAGAS...")
    print(f"   LLM для метрик: {RAGAS_LLM_MODEL}")
    print("   Метрики: Faithfulness, Context Precision")
    print("   (Answer Relevancy отключена — требует отдельной embeddings-настройки)\n")

    ragas_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ragas_llm = llm_factory(RAGAS_LLM_MODEL, client=ragas_client)
    metrics_to_use = [Faithfulness(), ContextPrecision()]

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            llm=ragas_llm,
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)

    faithfulness_values = result["faithfulness"]
    context_precision_values = result["context_precision"]
    avg_faithfulness = _safe_mean(faithfulness_values)
    avg_context_precision = _safe_mean(context_precision_values)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")

    avg_score = (avg_faithfulness + avg_context_precision) / 2
    print(f"\n{'─'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")

    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
    else:
        print("   Оценка: Требует значительного улучшения [X]")

    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)

    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\n{i + 1}. {question}")
        faith_val = faithfulness_values[i]
        cp_val = context_precision_values[i]
        if not (isinstance(faith_val, float) and math.isnan(faith_val)):
            print(f"   Faithfulness:       {faith_val:.4f}")
        else:
            print("   Faithfulness:       не удалось вычислить")
        if not (isinstance(cp_val, float) and math.isnan(cp_val)):
            print(f"   Context Precision:  {cp_val:.4f}")
        else:
            print("   Context Precision:  не удалось вычислить")

    print("\n" + "=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()
