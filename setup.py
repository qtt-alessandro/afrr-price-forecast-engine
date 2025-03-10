from setuptools import setup, find_packages

setup(
    name="energy_forecasting",
    version="0.1.0",
    description="Energy forecasting and optimization system",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(include=[
        "afrr_price_ts_forecast",
        "afrr_price_ts_forecast.*",
        "data_collection_module",
        "data_collection_module.*",
        "utils",
        "utils.*"
    ]),
    python_requires=">=3.8",
)