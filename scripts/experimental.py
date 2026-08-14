from dotenv import load_dotenv

from chunking_metrics.preparations import local

load_dotenv()

value = local.calculate_perplexity(
    "Он подписал договор на следующий день.",
    device="cpu",
)
print(f"perplexity={value:.6f}")
