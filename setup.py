from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

with open("requirements.txt", 'r', encoding='utf-8') as f:
    requirements = f.read()

setup(
    name="shap-e",
    author="OpenAI",
    install_requires=requirements,
    license='MIT License',
    description='This is the official code and model release for Shap-E: Generating Conditional 3D Implicit Functions.',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/openai/shap-e',
    packages=find_packages(),

)
