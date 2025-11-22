from setuptools import setup, find_packages

setup(
    name="stockdata",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'MetaTrader5',
        'pytz',
        'python-dotenv',
        'requests',
        'beautifulsoup4',
        'python-telegram-bot',
        'fastapi',
        'uvicorn',
        'backoff',
        'python-multipart',
        'python-jose[cryptography]',
        'passlib[bcrypt]',
    ],
    entry_points={
        'console_scripts': [
            'stockdata=STOCKDATA.main:main',
        ],
    },
    python_requires='>=3.8',
)
