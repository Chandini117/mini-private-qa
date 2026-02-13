# AI Usage Notes

AI tools were used during development for:

- Initial project structure setup
- Refining chunking logic for better retrieval
- Improving the grounding prompt for Gemini
- Debugging Hugging Face file handling behavior
- Improving system status checks

All generated code was reviewed and manually verified.

I tested:

- Retrieval accuracy using controlled test sentences
- Proper grounding behavior ("I don't know" when context missing)
- Workspace reset logic
- Empty input handling
- No API keys committed to repository

LLM Used:
- Gemini 2.5 Flash (Google Generative AI)

Reason for choosing Gemini:
- Fast inference speed
- Strong instruction-following capability
- Good contextual grounding when constrained