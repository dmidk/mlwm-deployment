# Development notes

## Local development

- currently image build only works on amd64 machines (i.e. not on macos)

- image build requires `aws` cli which can be retrieved from https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip

- to load AWS crendentials from `.aws/credentials` you can use the following script (drop in e.g. `~/.bashrc`):

```bash
aws-load-creds() {
  local profile=$1
  if [[ -z "$profile" ]]; then
    echo "❌ Usage: aws-load-creds <profile-name>"
    return 1
  fi

  local access_key
  local secret_key

  access_key=$(aws configure get aws_access_key_id --profile "$profile" 2>/dev/null)
  secret_key=$(aws configure get aws_secret_access_key --profile "$profile" 2>/dev/null)

  if [[ -z "$access_key" || -z "$secret_key" ]]; then
    echo "❌ The config profile '$profile' could not be found or is incomplete."
    return 1
  fi

  export AWS_ACCESS_KEY_ID="$access_key"
  export AWS_SECRET_ACCESS_KEY="$secret_key"

  echo "✅ Loaded AWS credentials from profile: $profile"
}

aws-list-profiles() {
  echo "📂 AWS profiles found:"
  grep '^\[profile ' ~/.aws/config 2>/dev/null | sed 's/^\[profile //' | sed 's/\]//'
  grep '^\[' ~/.aws/credentials 2>/dev/null | sed 's/^\[//' | sed 's/\]//'
}
```

- to set the environment variables for `./entry.sh` you can use a `.env` file. E.g. to run with DINI forecast data you would use:

```bash
# .env
ANALYSIS_TIME="2025-09-22T120000Z"
DINI_ZARR="s3://harmonie-zarr/dini/control/${ANALYSIS_TIME}/single_levels.zarr/"
DATASTORE_INPUT_PATHS="danra.danra_surface=${DINI_ZARR},danra.danra_static=${DINI_ZARR}"
TIME_DIMENSIONS="time"
```
