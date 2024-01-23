## How to start the server?

This is a Python 3.11 FastAPI project with the necessary requirements added to the `requirements.txt`.

### How to install Python 3.11?

[Python 3.11](https://www.python.org/downloads/release/python-3112/) installation should be pretty simple. Use your favorite search engine to figure out how to install Python 3.11. Note down the path of your installation.

### Create a virtual environment for your search engine project.

Use the following command to create a virtual environment specifically for SI 650.
Navigate to your search engine project folder and then use the following command:

```
<path of your python installation>/python -m venv si650
```

If you installed Python 3.11 without overwriting the default Python version, use the following command:

```
python3.11 -m venv si650
```

This will create a folder `si650`.

To activate the environment 

- For Windows
```
si650/bin/activate
```
- For others
```
source si650/bin/activate
```

### Install the requirements

After activating the virtual environment, run

```
python -m pip install -r requirements.txt
```
This should install the libraries you need to start the server.

### The dataset

1. The dataset should be in JSONL format where each lines is a separate JSON of the following format.

```
{
    "docid": <document id>
    "title": <document title>
    "text": <the entire text of the document>
    "categories": [<each Wikipedia category>]
}
```
**The name of the index folder to be generated is set in the `pipeline.py` file.**

After you have all of these files and the necessary Python requirements installed in your environment, run 

```
python -m uvicorn app:app
```

to start the server.

More comments are present in the code files themselves. And if you have trouble understanding parts of the code, please ping any of the GSIs for the course on Slack or create a post on Piazza.
