# News Agent

News Agent is a Python project for generating blog posts based on news for moms and images for them.

## Preparation
### API keys

This project involves API keys for NewsAPI to retrieve latest news and for Mistral to summarize them.

Since API keys are personal information, user has to get their own keys and specify them in *api_keys.json* file like so:

```json
{
    "NEWS_API_KEY": "<your key here>",
    "MISTRAL_API_KEY": "<your key here>"
}
```
If no keys are specified, **deep learning models** will be loaded and run locally. Note that running models locally (especially for image generation) requires a lot of free space and takes a lot of time.

### Requirements

Running this project requires python installed.
Packages for this project are specified in *requirements.txt*. You can install necessary packages via

```
!pip install -r requirements.txt
```
It's better to use IDE or code editors like VS Code to run this project.
