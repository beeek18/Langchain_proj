# scraper.py
import asyncio
import random
import re
from playwright.async_api import async_playwright
from langchain_core.documents import Document
from uuid import UUID


BASE_URL = "https://www.povarenok.ru"


def recipe_id_to_uuid(recipe_id: int) -> str:
    return str(UUID(int=recipe_id))


def url_to_id(url: str) -> int:
    return int(url.rstrip("/").split("/")[-1])

def parse_minutes(cook_time: str) -> int | None:
    hours = re.search(r"(\d+)\s*час", cook_time)
    mins = re.search(r"(\d+)\s*мин", cook_time)
    total = 0
    if hours: total += int(hours.group(1)) * 60
    if mins: total += int(mins.group(1))
    return total if total > 0 else None


async def get_links(query: str, limit: int = 10) -> list[str]:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="ru-RU",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        await page.goto(f"{BASE_URL}/recipes/poisk/{query}/?orderby=rating")
        await asyncio.sleep(2)
        anchors = await page.query_selector_all("article.item-bl h2 a")
        links = []
        for a in anchors[:limit]:
            href = await a.get_attribute("href") or ""
            if "/recipes/show/" in href:
                links.append(href)
        await browser.close()
        return links


async def parse_page(page, url: str) -> dict:
    await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    await asyncio.sleep(random.uniform(0.8, 1.5))
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
    await asyncio.sleep(0.5)
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await asyncio.sleep(0.5)

    title_el = await page.query_selector("h1")
    title = (await title_el.inner_text()).strip() if title_el else ""

    crumbs = await page.query_selector_all("[itemprop='recipeCategory'] a")
    categories = [(await c.inner_text()).strip() for c in crumbs]

    cuisine_el = await page.query_selector("[itemprop='recipeCuisine'] a")
    cuisine = (await cuisine_el.inner_text()).strip() if cuisine_el else ""

    time_el = await page.query_selector("time[itemprop='totalTime']")
    cook_time = (await time_el.inner_text()).strip() if time_el else ""

    ingredients = []
    for el in await page.query_selector_all("li[itemprop='recipeIngredient']"):
        span = await el.query_selector("a span")
        if span:
            t = (await span.inner_text()).strip()
            if t:
                ingredients.append(t)

    destiny, tags, tastes = [], [], []
    for p in await page.query_selector_all(".tab-content p"):
        text = await p.inner_text()
        values = [(await a.inner_text()).strip() for a in await p.query_selector_all("a")]
        if "Назначение" in text:
            destiny = values
        elif "Теги" in text:
            tags = values
        elif "Вкусы" in text:
            tastes = values

    per100_kcal, per100_protein, per100_fat, per100_carbs = "", "", "", ""
    full_text = (await page.evaluate("document.body.innerText")).lower().replace("\xa0", " ")
    idx = full_text.find("100 г блюда")
    if idx != -1:
        zone = full_text[idx:idx+200]
        m = re.search(r"([\d.,]+)\s*ккал", zone)
        if m: per100_kcal = m.group(1)
        for key, field in [("белки", "per100_protein"), ("жиры", "per100_fat"), ("углеводы", "per100_carbs")]:
            m = re.search(rf"{key}\s+([\d.,]+)", zone)
            if m:
                if field == "per100_protein": per100_protein = m.group(1)
                elif field == "per100_fat": per100_fat = m.group(1)
                elif field == "per100_carbs": per100_carbs = m.group(1)

    return {
        "id": url_to_id(url),
        "title": title,
        "categories": categories,
        "cuisine": cuisine,
        "ingredients": ingredients,
        "cook_time_minutes": parse_minutes(cook_time),
        "per100_kcal": float(per100_kcal.replace(",", ".")) if per100_kcal else None,   
        "per100_protein": float(per100_protein.replace(",", ".")) if per100_protein else None, 
        "per100_fat": float(per100_fat.replace(",", ".")) if per100_fat else None,      
        "per100_carbs": float(per100_carbs.replace(",", ".")) if per100_carbs else None,
        "destiny": destiny,
        "tags": tags,
        "tastes": tastes,
        "url": url,
    }


async def save_to_qdrant(url: str, context, vector_store) -> dict | None:
    page = await context.new_page()
    try:
        recipe = await parse_page(page, url)
        page_content = (
            f"{recipe['title']}. "
            f"Категория: {', '.join(recipe['categories'])}. "
            f"Ингредиенты: {', '.join(recipe['ingredients'])}. "
            f"Назначение: {', '.join(recipe['destiny'])}. "
            f"Теги: {', '.join(recipe['tags'])}. "
            f"Вкусы: {', '.join(recipe['tastes'])}."
        )
        doc = Document(page_content=page_content, metadata=recipe)
        vector_store.add_documents([doc], ids=[recipe_id_to_uuid(recipe["id"])])
        print(f"  ✅ {recipe['title']}")
        return recipe
    except Exception as e:
        print(f"  ❌ {url}: {e}")
        return None
    finally:
        await page.close()


async def scrape(query: str, vector_store, limit: int = 10) -> list[dict]:
    print(f"🔍 Ищем: '{query}'")
    links = await get_links(query, limit=limit)
    print(f"   Найдено ссылок: {len(links)}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="ru-RU",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
        )

        results = await asyncio.gather(*[
            save_to_qdrant(url, context, vector_store)
            for url in links
        ])

        await browser.close()

    recipes = [r for r in results if r is not None]
    print(f"✅ Сохранено: {len(recipes)}")
    return recipes