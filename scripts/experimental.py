from dotenv import load_dotenv

from chunking_metrics.preparation import calculate_perplexity

load_dotenv()

value = calculate_perplexity(
    "Он подписал договор на следующий день.",
    device="cpu",
)
print(f"perplexity={value:.6f}")
