# tus-ia-team-project

Engineering Team Project for IA Master

### Configuration

#### Setting up config.json

1. Create a file named config.json in the root directory of the project.
2. Open config.json and structure it as follows:

```json
{
  "twitter": {
    "consumer_key": "YOUR_CONSUMER_KEY",
    "consumer_secret": "YOUR_CONSUMER_SECRET",
    "access_token": "YOUR_ACCESS_TOKEN",
    "access_token_secret": "YOUR_ACCESS_TOKEN_SECRET"
  }
}
```

3. Replace `YOUR_CONSUMER_KEY`, `YOUR_CONSUMER_SECRET`, `YOUR_ACCESS_TOKEN`, and `YOUR_ACCESS_TOKEN_SECRET` with your actual Twitter API credentials.
   **Note**: `config.json` is included in the .gitignore file to prevent accidental exposure of sensitive information.
