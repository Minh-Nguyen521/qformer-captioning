from setuptools import setup, find_packages

setup(
    name="qformer-captioning",
    version="0.1.0",
    description="BLIP-2-style image captioning with Q-Former (CLIP ViT + Flan-T5)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "metrics": [
            "nltk>=3.8",
            "rouge-score>=0.1.2",
        ],
    },
)
