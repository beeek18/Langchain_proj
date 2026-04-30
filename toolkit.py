# toolkit.py
import nest_asyncio
nest_asyncio.apply()

import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from scraper import scrape, recipe_id_to_uuid


class SearchRecipesInput(BaseModel):
    query: str = Field(description="Запрос пользователя для поиска рецепта на русском, например 'борщ', 'что-нибудь сладкое', 'лёгкий ужин'")
    exclude_ingredients: list[str] = Field(default=[], description="Ингредиенты которые нужно исключить на русском, например ['яйца', 'глютен', 'молоко']")
    k: int = Field(default=3, description="Количество результатов, от 1 до 5")


class FindSimilarInput(BaseModel):
    recipe_id: int = Field(description="ID рецепта из поля metadata.id, например 1001 или 25436")
    limit: int = Field(default=3, description="Количество похожих рецептов, от 1 до 5")


class ScrapeAndSaveInput(BaseModel):
    query: str = Field(description="Название блюда транслитом для поиска на povarenok.ru, например 'borsch', 'plov', 'pelmeni'")
    limit: int = Field(default=10, description="Количество рецептов для скрапинга, от 1 до 10")
class RecipeToolkit:
    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store
        self.client = vector_store.client
        self.collection_name = vector_store.collection_name

    def get_tools(self):

        @tool("search_recipes", args_schema=SearchRecipesInput)
        def search_recipes(query: str, exclude_ingredients: list[str] = [], k: int = 3) -> str:
            """Ищет рецепты в базе данных по смыслу запроса. Может исключать рецепты с определёнными ингредиентами."""
            results = self.vector_store.similarity_search(query, k=k+2)

            if not results:
                return "Ничего не найдено."

            if exclude_ingredients:
                results = [
                    r for r in results
                    if not any(
                        exc.lower() in [i.lower() for i in r.metadata["ingredients"]]
                        for exc in exclude_ingredients
                    )
                ]

            if not results:
                return f"Рецепты найдены, но все содержат: {', '.join(exclude_ingredients)}."

            output = []
            for r in results[:k]:
                m = r.metadata
                output.append(
                    f"=== {m['title']} ===\n"
                    f"ID: {m['id']}\n"
                    f"Категории: {', '.join(m['categories'])}\n"
                    f"Время: {m['cook_time']}\n"
                    f"Ингредиенты: {', '.join(m['ingredients'])}\n"
                    f"Назначение: {', '.join(m['destiny'])}\n"
                    f"Вкусы: {', '.join(m['tastes'])}\n"
                    f"Калории (100г): {m['per100_kcal']} ккал\n"
                    f"Ссылка: {m['url']}\n"
                )

            return "\n".join(output)

        @tool("find_similar", args_schema=FindSimilarInput)
        def find_similar(recipe_id: int, limit: int = 3) -> str:
            """Находит похожие рецепты по ID существующего рецепта."""
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

            output = []
            for r in results:
                m = r.payload["metadata"]
                output.append(
                    f"=== {m['title']} ===\n"
                    f"ID: {m['id']}\n"
                    f"Категории: {', '.join(m['categories'])}\n"
                    f"Время: {m['cook_time']}\n"
                    f"Вкусы: {', '.join(m['tastes'])}\n"
                    f"Ссылка: {m['url']}\n"
                )

            return "\n".join(output)

        @tool("scrape_and_save_recipe", args_schema=ScrapeAndSaveInput)
        def scrape_and_save_recipe(query: str, limit: int = 10) -> str:
            """Ищет рецепты на povarenok.ru по запросу, парсит и сохраняет в базу данных."""
            recipes = asyncio.run(scrape(query, self.vector_store, limit=limit))

            if not recipes:
                return f"Не удалось найти рецепты по запросу '{query}'."

            titles = [r["title"] for r in recipes]
            return f"Нашёл и сохранил {len(recipes)} рецептов: {', '.join(titles)}"

        return [search_recipes, find_similar, scrape_and_save_recipe]