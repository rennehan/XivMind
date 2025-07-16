from XivMind.pipeline import Pipeline
import asyncio

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    pipeline = Pipeline("OpenAI:gpt-3.5-turbo", "OpenAI:text-embedding-3-small")

    # pipeline.request_model_from_user()

    pipeline.load_agents()

    while True:
        asyncio.run(pipeline.query_and_respond())

        yn = input("Do you want to ask another question? (y/n): ")
        if yn.lower() != "y":
            break


