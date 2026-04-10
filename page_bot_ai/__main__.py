"""Entry point: python -m bot_page_ai"""

from .flow import run_chatbot

answer = run_chatbot(
    question="How to install Cypress?",
    target_urls=["https://docs.cypress.io/app/get-started/why-cypress"],
    instruction="Provide clear, beginner-friendly explanations with examples.",
)

print("\n" + "=" * 60)
print("FINAL ANSWER:")
print("=" * 60)
print(answer)
