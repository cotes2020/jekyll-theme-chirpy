# Programming With Wolfgang

A tech blog focusing on DevOps, Cloud, Azure, Kubernetes and Software Architecture.

## Checkout

Checkout the repository on Linux or WSL2 if you are on Windows. 

## Setting up the local envrionment

The whole environment is built and run inside a Docker container. To run the solution, execute the following command:

```terminal
docker run --volume="$($PWD):/srv/jekyll" -p 4000:4000 -it jekyll/jekyll bash tools/run.sh
```
The run script creates all tags and posts inside a category. If running the sh script does not work, execute the following command:

```terminal
dos2unix tools/run.sh
dos2unix tools/build.sh
```

If you only want to run the website, you can use the following code, which should start faster than the above one:

```terminal
docker run --volume="$($PWD):/srv/jekyll" -p 4000:4000 -it jekyll/jekyll jekyll serve --force_polling
```
The --force_polling flag enables a watcher that re-creates the files everytime something changes.

## Setting up the live environment

Currently, the website needs to be built and the _site folder needs to be checked in. Build the site with the following command:

```terminal
docker run --rm --volume="$($PWD):/srv/jekyll" -p 4000:4000 -it jekyll/jekyll bash tools/run.sh --build
```

It is planned to move this task to the Github action in the near future.
