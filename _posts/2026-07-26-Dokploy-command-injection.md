---
title: Authenticated Command Injection in Dokploy Dockerfile Builder
date: 2026-4-11 13:10:30 +0000
categories: [Security Research, CVES]
tags: [Security Research, CVES]
math: true
mermaid: true
media_subpath: /assets/posts/cves/dokploy
image:
  path: preview.jpeg
---
### Summary
This Dokploy dockerfile builder is vulnerable to command injection due to unfiltered input `dockerContextPath` being passed into a shell statement leading to command injection.

### Details
- Affected Code in docker-file builder `packages/server/src/utils/builders/docker-file.ts`
- In line 87, the variable dockerContextPath gets passed into the command for building an image with a dockerfile.This command is later executed with the aid of  `execAsync()` allowing shell command execution.

```typescript
//line 87
command += `
echo "Building ${appName}" ;
cd ${dockerContextPath} || { 
  echo "❌ The path ${dockerContextPath} does not exist" ;
  exit 1;
}

${joinedSecrets} docker ${commandArgs.join(" ")} || { 
  echo "❌ Docker build failed" ;
  exit 1;
}
echo "✅ Docker build completed." ;
```
-  The variable `dockerContextPath` is unfiltered.Let’s trace the flow. It starts from here-: 

```typescript
//Line 32
const dockerContextPath =
			getDockerContextPath(application) || defaultContextPath;

```
- The method `getDockerContextPath` is a function in `packages/server/src/utils/filesystem/directory.ts`. It grabs the key `dockerContextPath` from  application and adds it to a path without filtering the characters.

```typescript
export const getDockerContextPath = (application: Application) => {
	const { APPLICATIONS_PATH } = paths(!!application.serverId);
	const { appName, dockerContextPath } = application;

	if (!dockerContextPath) {
		return null;
	}
	
return path.join(APPLICATIONS_PATH, appName, "code",dockerContextPath);
```

- `dockerContextPath` is unfiltered and accepts strings  with special characters as seen in the schema in apps/dokploy/components/dashboard/application/build/show.tsx.

```typescript
// Line 83
dockerContextPath: z.string().nullable().default("")
```

### PoC
- Upload a zip file archive with a valid dockerfile
- Set the dockerContextPath with a payload like this “default$(echo `id`)” to echo the output in the log.

   
![image](552123332-6f25d263-a6a7-45b1-8d4c-cc9d625380e9.png)

-  Deploy it and check “Deployment” tab to see the value of the payload

![image](552123625-f3a6d74e-59d1-4090-a807-f593c8694653.png)



### Impact
- It allows an authenticated attacker to gain remote code execution on a Dokploy server leading to unauthorized access to a server.

- [Original Advisory](https://github.com/Dokploy/dokploy/security/advisories/GHSA-qjrc-g63x-qhp9)