# smart_healthcare_flask

Step 1:
After cloning the repo, using powershell, create an environment in the root folder. insert whatever environment name u want

```virtualenv -p python {env-name}```

```[env-name]/Scripts/activate```

Step 2:
After entering env, under main, run

``` pip install -r requirements.txt ```

Step 3:
It take a long time to download, because of keras, tensorflow. run

```$env:FLASK_APP = "app"```

```flask run```
