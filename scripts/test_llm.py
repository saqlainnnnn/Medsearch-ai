from app.llm.cerebras import CerebrasLLM


def main() -> None:
    llm = CerebrasLLM()

    response = llm.generate(
        """
        What is diabetes?
        Answer in 3 sentences.
        """
    )

    print(response)


if __name__ == "__main__":
    main()