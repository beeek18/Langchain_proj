# toolkit.py
import nest_asyncio
nest_asyncio.apply()

import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, Range
from scraper import scrape, recipe_id_to_uuid


class SearchRecipesInput(BaseModel):
    query: str = Field(description="Название или описание блюда на русском, например 'плов', 'борщ', 'что-нибудь сладкое'")
    k: int = Field(default=3, description="Количество результатов от 1 до 5, по умолчанию 3")


class SearchRecipesWithFiltersInput(BaseModel):
    query: str = Field(description="Название или описание блюда на русском, например 'плов', 'борщ'")
    exclude_ingredients: list[str] = Field(default=[], description="Корни слов ингредиентов на русском которые нужно исключить при аллергии или диете, например ['яйц', 'молок', 'свинин', 'глютен']")
    min_cook_time: int | None = Field(default=None, description="Минимальное время готовки в минутах, целое число, например 30.")
    max_cook_time: int | None = Field(default=None, description="Максимальное время готовки в минутах, целое число, например 60. Используй если пользователь хочет быстрый рецепт.")
    min_calories: float | None = Field(default=None, description="Минимум калорий на 100г блюда, например 100.0.")
    max_calories: float | None = Field(default=None, description="Максимум калорий на 100г блюда, например 300.0. Используй если пользователь хочет диетическое блюдо.")
    min_protein: float | None = Field(default=None, description="Минимум белков на 100г блюда, например 10.0. Используй если пользователь хочет высокобелковое блюдо.")
    max_protein: float | None = Field(default=None, description="Максимум белков на 100г блюда, например 20.0.")
    min_fat: float | None = Field(default=None, description="Минимум жиров на 100г блюда, например 5.0.")
    max_fat: float | None = Field(default=None, description="Максимум жиров на 100г блюда, например 15.0. Используй если пользователь хочет нежирное блюдо.")
    min_carbs: float | None = Field(default=None, description="Минимум углеводов на 100г блюда, например 10.0.")
    max_carbs: float | None = Field(default=None, description="Максимум углеводов на 100г блюда, например 30.0. Используй если пользователь на кето или низкоуглеводной диете.")
    k: int = Field(default=3, description="Количество результатов от 1 до 5, по умолчанию 3")


class FindSimilarInput(BaseModel):
    recipe_id: int = Field(description="ID рецепта из поля metadata.id, например 25436. Берётся из результатов search_recipes.")
    limit: int = Field(default=3, description="Количество похожих рецептов от 1 до 5, по умолчанию 3")


class ScrapeAndSaveInput(BaseModel):
    query: str = Field(description="Только название блюда транслитом, одно-два слова, например 'borsch', 'plov', 'pelmeni', 'shokoladnyj-tort'. Не передавай полные фразы.")
    limit: int = Field(default=10, description="Количество рецептов для скрапинга от 1 до 10, по умолчанию 10")


def format_recipe(m: dict) -> str:
    return (
        f"=== {m.get('title', '—')} ===\n"
        f"ID: {m.get('id', '—')}\n"
        f"Категории: {', '.join(m.get('categories', []))}\n"
        f"Время: {str(m.get('cook_time_minutes')) + ' мин' if m.get('cook_time_minutes') else '—'}\n"
        f"Ингредиенты: {', '.join(m.get('ingredients', []))}\n"
        f"Назначение: {', '.join(m.get('destiny', []))}\n"
        f"Вкусы: {', '.join(m.get('tastes', []))}\n"
        f"Калории (100г): {m.get('per100_kcal', '—')} ккал\n"
        f"Ссылка: {m.get('url', '—')}\n"
    )


class RecipeToolkit:
    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store
        self.client = vector_store.client
        self.collection_name = vector_store.collection_name

    def get_tools(self):

        @tool("search_recipes", args_schema=SearchRecipesInput)
        def search_recipes(query: str, k: int = 3) -> str:
            """Ищет рецепты в базе данных БЕЗ фильтров.
            Используй ПЕРВЫМ при любом запросе о еде — только название блюда.
            Если нужное блюдо найдено И есть фильтры — используй search_recipes_with_filters."""
            results = self.vector_store.similarity_search(query, k=k)
            if not results:
                return "Ничего не найдено."
            return "\n".join(format_recipe(r.metadata) for r in results)

        @tool("search_recipes_with_filters", args_schema=SearchRecipesWithFiltersInput)
        def search_recipes_with_filters(
            query: str,
            exclude_ingredients: list[str] = [],
            min_cook_time: int | None = None,
            max_cook_time: int | None = None,
            min_calories: float | None = None,
            max_calories: float | None = None,
            min_protein: float | None = None,
            max_protein: float | None = None,
            min_fat: float | None = None,
            max_fat: float | None = None,
            min_carbs: float | None = None,
            max_carbs: float | None = None,
            k: int = 3,
        ) -> str:
            """Ищет рецепты С фильтрами по времени, калориям, БЖУ и аллергенам.
            Используй ТОЛЬКО после search_recipes когда убедился что блюдо есть в базе."""
            filters = [
                ("metadata.cook_time_minutes", min_cook_time, max_cook_time),
                ("metadata.per100_kcal",       min_calories,  max_calories),
                ("metadata.per100_protein",    min_protein,   max_protein),
                ("metadata.per100_fat",        min_fat,       max_fat),
                ("metadata.per100_carbs",      min_carbs,     max_carbs),
            ]
            must = [
                FieldCondition(key=key, range=Range(gte=min_val, lte=max_val))
                for key, min_val, max_val in filters
                if min_val is not None or max_val is not None
            ]
            qdrant_filter = Filter(must=must) if must else None
            results = self.vector_store.similarity_search(query, k=k+5, filter=qdrant_filter)

            if not results:
                return (
                    "По заданным фильтрам ничего не найдено. "
                    "Блюдо есть в базе, но не соответствует фильтрам. "
                    "Сообщи пользователю об этом и предложи ослабить ограничения."
                )

            if exclude_ingredients:
                results = [
                    r for r in results
                    if not any(
                        exc.lower() in i.lower()
                        for exc in exclude_ingredients
                        for i in r.metadata.get("ingredients", [])
                    )
                ]

            if not results:
                return "Ничего не найдено по заданным фильтрам."

            return "\n".join(format_recipe(r.metadata) for r in results[:k])

        @tool("find_similar", args_schema=FindSimilarInput)
        def find_similar(recipe_id: int, limit: int = 3) -> str:
            """Находит похожие рецепты по ID существующего рецепта.
            Используй когда пользователь хочет найти что-то похожее на конкретный рецепт
            и у тебя уже есть его ID из предыдущего поиска."""
            uuid = recipe_id_to_uuid(recipe_id)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=uuid,
                limit=limit+1,
                with_payload=True,
            ).points
            results = [r for r in results if r.payload["metadata"]["id"] != recipe_id]

            if not results:
                return "Похожие рецепты не найдены."

            return "\n".join(format_recipe(r.payload["metadata"]) for r in results)

        @tool("scrape_and_save_recipe", args_schema=ScrapeAndSaveInput)
        def scrape_and_save_recipe(query: str, limit: int = 10) -> str:
            """Ищет рецепты на povarenok.ru, парсит и сохраняет в базу данных.
            Используй ТОЛЬКО если search_recipes не нашёл нужного блюда вообще.
            Принимает название блюда транслитом: 'borsch', 'plov', 'pelmeni'."""
            recipes = asyncio.run(scrape(query, self.vector_store, limit=limit))
            if not recipes:
                return f"Не удалось найти рецепты по запросу '{query}'."

            output = [f"Нашёл и сохранил {len(recipes)} рецептов. Вот лучшие:\n"]
            for r in recipes:
                output.append(format_recipe(r))
            return "\n".join(output)

        return [search_recipes, search_recipes_with_filters, find_similar, scrape_and_save_recipe]