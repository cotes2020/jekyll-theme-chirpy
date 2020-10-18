---
title: Create a .NET Core Visual Studio Template
date: 2020-07-27T19:18:58+02:00
author: Wolfgang Ofner
categories: [Programming, Tools]
tags: [.net core 3.1,, 'C#', docker, Visual Studio]
---
Over the last couple of weeks, I created several microservices with the same project structure. Setting up a new solution was quite repetitive, time-consuming, and no to be honest no fun. Since my solutions have the same structure, I created a Visual Studio template that sets up everything for me. Today, I want to show you how to create your Visual Studio template with ASP .Net MVC with .NET Core 3.1 and Docker.

## Creating a Visual Studio Template

I plan to create a skeleton for my solution and when creating a new project with my template to take the user input as the name for the projects and the entities. For example, if the user names the solution Customer, I will create a Customer.Api, Customer.Service and Customer.Data project and also a Customer entity. You can find the code of the demo on <a href="https://github.com/WolfgangOfner/VisualStudioTemplate" target="_blank" rel="noopener noreferrer">GitHub</a>. Note that there are two branches.

### Creating the Structure of the Project for the Visual Studio Template

The first step is to create the structure of your Visual Studio template. Therefore, I created an empty C# solution and add three projects, Template.Api, Template.Service and Template.Data. For the template, you can install any Nuget you want or add code. I added a controller that calls a service that calls a repository to get some data.

<div id="attachment_2285" style="width: 271px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Visual-Studio-Template-Project-Structure.jpg"><img aria-describedby="caption-attachment-2285" loading="lazy" class="size-full wp-image-2285" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Visual-Studio-Template-Project-Structure.jpg" alt="Visual Studio Template Project Structure" width="261" height="365" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Visual-Studio-Template-Project-Structure.jpg 261w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Visual-Studio-Template-Project-Structure-215x300.jpg 215w" sizes="(max-width: 261px) 100vw, 261px" /></a>
  
  <p id="caption-attachment-2285" class="wp-caption-text">
    Visual Studio Template Project Structure
  </p>
</div>

Additionally, I added Swagger to have a UI to test the Api method. When you are satisfied with your project, run it. You should see the Swagger UI now.

<div id="attachment_2286" style="width: 378px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-Swagger-UI-for-the-template.jpg"><img aria-describedby="caption-attachment-2286" loading="lazy" class="size-full wp-image-2286" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-Swagger-UI-for-the-template.jpg" alt="The Swagger UI for the template" width="368" height="426" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-Swagger-UI-for-the-template.jpg 368w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-Swagger-UI-for-the-template-259x300.jpg 259w" sizes="(max-width: 368px) 100vw, 368px" /></a>
  
  <p id="caption-attachment-2286" class="wp-caption-text">
    The Swagger UI for the template
  </p>
</div>

As the last step, I test the Get method to verify that everything works correctly.

<div id="attachment_2287" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Testing-the-Template-Api.jpg"><img aria-describedby="caption-attachment-2287" loading="lazy" class="wp-image-2287" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Testing-the-Template-Api.jpg" alt="Testing the Template Api" width="700" height="712" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Testing-the-Template-Api.jpg 750w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Testing-the-Template-Api-295x300.jpg 295w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2287" class="wp-caption-text">
    Testing the Template Api
  </p>
</div>

The template is set up and as the next step, I am going to export it and install it in Visual Studio.

### Export the Template

To export the Visual Studio Template, click on Project and then Export Template.

<div id="attachment_2288" style="width: 361px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Export-the-Visual-Studio-Template.jpg"><img aria-describedby="caption-attachment-2288" loading="lazy" class="size-full wp-image-2288" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Export-the-Visual-Studio-Template.jpg" alt="Export the Visual Studio Template" width="351" height="548" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Export-the-Visual-Studio-Template.jpg 351w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Export-the-Visual-Studio-Template-192x300.jpg 192w" sizes="(max-width: 351px) 100vw, 351px" /></a>
  
  <p id="caption-attachment-2288" class="wp-caption-text">
    Export the Visual Studio Template
  </p>
</div>

This opens a new window in which you can select what template you want to export and which project. Leave it as Project template, select the Template.Api project and click on Next.

<div id="attachment_2289" style="width: 725px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Choose-which-Project-to-export.jpg"><img aria-describedby="caption-attachment-2289" loading="lazy" class="size-full wp-image-2289" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Choose-which-Project-to-export.jpg" alt="Choose which Project to export" width="715" height="566" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Choose-which-Project-to-export.jpg 715w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Choose-which-Project-to-export-300x237.jpg 300w" sizes="(max-width: 715px) 100vw, 715px" /></a>
  
  <p id="caption-attachment-2289" class="wp-caption-text">
    Choose which Project to export
  </p>
</div>

On the last page of the export, uncheck both checkboxes and click Finish.

<div id="attachment_2290" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Finish-the-Export.jpg"><img aria-describedby="caption-attachment-2290" loading="lazy" class="wp-image-2290" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Finish-the-Export.jpg" alt="Finish the Export" width="700" height="556" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Finish-the-Export.jpg 715w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Finish-the-Export-300x238.jpg 300w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2290" class="wp-caption-text">
    Finish the Export
  </p>
</div>

Repeat this export for all other projects in your solution. After you are finished, you can find the exported files under C:\Users\<YourUserName>\Documents\Visual Studio <YourVersion>\My Exported Templates.

<div id="attachment_2291" style="width: 629px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-exported-.zip-files.jpg"><img aria-describedby="caption-attachment-2291" loading="lazy" class="size-full wp-image-2291" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-exported-.zip-files.jpg" alt="The exported .zip files" width="619" height="140" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-exported-.zip-files.jpg 619w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-exported-.zip-files-300x68.jpg 300w" sizes="(max-width: 619px) 100vw, 619px" /></a>
  
  <p id="caption-attachment-2291" class="wp-caption-text">
    The exported .zip files
  </p>
</div>

Unzip every zip file into a separate folder and delete the zip files. I get quite often a warning that the file header is corrupt during the unzip. You can ignore this message though. Next, create a file with a vstemplate ending, for example, Template.vstemplate in the folder where you unzipped your templates. This file contains links to all projects in the template in the XML format. Copy the following code into the file:

[code language=&#8221;XML&#8221;]  
<VSTemplate Version="2.0.0" Type="ProjectGroup"  
xmlns="http://schemas.microsoft.com/developer/vstemplate/2005">  
<TemplateData>  
<Name>My Template</Name>  
<Description>A template with a three tier architecture.</Description>  
<ProjectType>CSharp</ProjectType>  
</TemplateData>  
<TemplateContent>  
<ProjectCollection>  
<ProjectTemplateLink ProjectName=" Template.Api" CopyParameters="true">  
Template.Api\MyTemplate.vstemplate  
</ProjectTemplateLink>  
<ProjectTemplateLink ProjectName=" Template.Service" CopyParameters="true">  
Template.Service\MyTemplate.vstemplate  
</ProjectTemplateLink>  
<ProjectTemplateLink ProjectName="Template.Data" CopyParameters="true">  
Template.Data\MyTemplate.vstemplate  
</ProjectTemplateLink>  
</ProjectCollection>  
</TemplateContent>  
</VSTemplate>  
[/code]

Save the file and create a zip of the three folder and the template file. You can easily do this by highlighting everything and the right-click and then select Send to &#8211;> Compressed (zipped) folder. Your folder should contain the following files and directories now:

<div id="attachment_2292" style="width: 632px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Content-of-the-template-folder-after-finishing-the-creation-of-the-template.jpg"><img aria-describedby="caption-attachment-2292" loading="lazy" class="size-full wp-image-2292" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Content-of-the-template-folder-after-finishing-the-creation-of-the-template.jpg" alt="Content of the template folder after finishing the creation of the template" width="622" height="180" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Content-of-the-template-folder-after-finishing-the-creation-of-the-template.jpg 622w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Content-of-the-template-folder-after-finishing-the-creation-of-the-template-300x87.jpg 300w" sizes="(max-width: 622px) 100vw, 622px" /></a>
  
  <p id="caption-attachment-2292" class="wp-caption-text">
    Content of the template folder after finishing the creation of the template
  </p>
</div>

### Install the Visual Studio Template

To install the template, all you have to do is to copy the Template.zip file into the following folder: C:\Users\<YourNameUser>\Documents\Visual Studio <YourVersion>\Templates\ProjectTemplates\Visual C#. Alternatively, you could copy the file into the ProjectTemplates folder.

### Testing the new Template

Open Visual Studio and select Create a new project. Search for My Template and the previously added template will be displayed.

<div id="attachment_2293" style="width: 574px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Select-your-template-in-Visual-Studio.jpg"><img aria-describedby="caption-attachment-2293" loading="lazy" class="size-full wp-image-2293" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Select-your-template-in-Visual-Studio.jpg" alt="Select your template in Visual Studio" width="564" height="167" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Select-your-template-in-Visual-Studio.jpg 564w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Select-your-template-in-Visual-Studio-300x89.jpg 300w" sizes="(max-width: 564px) 100vw, 564px" /></a>
  
  <p id="caption-attachment-2293" class="wp-caption-text">
    Select your template in Visual Studio
  </p>
</div>

Create the project and you will see the same structure as in the template. When you start the project, you will see the Swagger UI and can test the Api methods.

## Make the Template flexible with Variables

Having a template is nice but it would be even nicer if the projects weren&#8217;t named Template and if they would take the name I provide. To achieve this, Visual Studio offers template parameters. You can find all available parameters <a href="https://docs.microsoft.com/en-us/visualstudio/ide/template-parameters?view=vs-2019" target="_blank" rel="noopener noreferrer">here</a>. The goal is to name the projects and files accordingly to the user input during the creation of the solution. To do this, I use the variable safeprojectname. This variable gives you the name of the current project. To get the name of the solution prefix it with ext\_, therefore I will use ext\_safeprojectname. All variables have a Dollar sign at the beginning and the end.

### Replace Class Names and Namespaces with Variables

I am replacing in all files Template with $ext_safeprojectname$, for example, the TemplateService class:

[code language=&#8221;CSharp&#8221;]  
using System.Collections.Generic;  
using $ext_safeprojectname$.Data;

namespace $ext_safeprojectname$.Service  
{  
public class $ext\_safeprojectname$Service : I$ext\_safeprojectname$Service  
{  
private readonly I$ext\_safeprojectname$Repository \_repository;

public TemplateService(I$ext_safeprojectname$Repository repository)  
{  
_repository = repository;  
}

public List<string> GetAll()  
{  
return _repository.GetAll();  
}  
}  
}  
[/code]

Adding the variable also adds a lot of errors in your solution. You can ignore them though.

<div id="attachment_2294" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Errors-after-adding-the-template-variables.jpg"><img aria-describedby="caption-attachment-2294" loading="lazy" class="wp-image-2294" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Errors-after-adding-the-template-variables.jpg" alt="Errors after adding the template variables" width="700" height="508" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Errors-after-adding-the-template-variables.jpg 720w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Errors-after-adding-the-template-variables-300x218.jpg 300w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2294" class="wp-caption-text">
    Errors after adding the template variables
  </p>
</div>

### Replace File Names with Variables

Not only class names and namespaces should have the provided name, but also the classes should be named accordingly. You can do the same as before and replace Template in all file names with $ext_safeprojectname$. You don&#8217;t have to change the project name though.

<div id="attachment_2295" style="width: 308px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Replace-the-file-names-with-variables.jpg"><img aria-describedby="caption-attachment-2295" loading="lazy" class="size-full wp-image-2295" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Replace-the-file-names-with-variables.jpg" alt="Replace the file names with variables" width="298" height="363" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Replace-the-file-names-with-variables.jpg 298w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Replace-the-file-names-with-variables-246x300.jpg 246w" sizes="(max-width: 298px) 100vw, 298px" /></a>
  
  <p id="caption-attachment-2295" class="wp-caption-text">
    Replace the file names with variables
  </p>
</div>

### Export the Template

Repeat the export from before by clicking on Project &#8211;> Export Template and export all your projects. Delete your previously created folders and unzipp the exported zip files. In the vstemplate file, replace Template with $safeprojectname$. This will rename the project files. Also make sure that CopyParameters=&#8221;true&#8221; is set for every project. Otherwise, the user input won&#8217;t be copied and the variables will be empty.

[code language=&#8221;XML&#8221;]  
<VSTemplate Version="2.0.0" Type="ProjectGroup"  
xmlns="http://schemas.microsoft.com/developer/vstemplate/2005">  
<TemplateData>  
<Name>My Template</Name>  
<Description>A template with a three tier architecture.</Description>  
<ProjectType>CSharp</ProjectType>  
</TemplateData>  
<TemplateContent>  
<ProjectCollection>  
<ProjectTemplateLink ProjectName="$safeprojectname$.Api" CopyParameters="true">  
Template.Api\MyTemplate.vstemplate  
</ProjectTemplateLink>  
<ProjectTemplateLink ProjectName="$safeprojectname$.Service" CopyParameters="true">  
Template.Service\MyTemplate.vstemplate  
</ProjectTemplateLink>  
<ProjectTemplateLink ProjectName="$safeprojectname$.Data" CopyParameters="true">  
Template.Data\MyTemplate.vstemplate  
</ProjectTemplateLink>  
</ProjectCollection>  
</TemplateContent>  
</VSTemplate>  
[/code]

Zip all files and copy the zip over the previously created zip in the Visual C# folder. Create a new project and select your template and enter Customer as project name. If you did everything right, all files and projects should be named correctly and the project should build without an error. It is very easy to have errors on the first try since you don&#8217;t have any help to find errors in the template and every typo will result in a build error.

<div id="attachment_2296" style="width: 257px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/All-files-got-renamed-correctly.jpg"><img aria-describedby="caption-attachment-2296" loading="lazy" class="wp-image-2296 size-full" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/All-files-got-renamed-correctly.jpg" alt="All files got renamed correctly from the Visual Studio Template" width="247" height="339" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/All-files-got-renamed-correctly.jpg 247w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/All-files-got-renamed-correctly-219x300.jpg 219w" sizes="(max-width: 247px) 100vw, 247px" /></a>
  
  <p id="caption-attachment-2296" class="wp-caption-text">
    All files got renamed correctly
  </p>
</div>

When you don&#8217;t have any error, run the project and you should see Customer in the headline, description and Controller.

<div id="attachment_2297" style="width: 372px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Testing-the-created-solution.jpg"><img aria-describedby="caption-attachment-2297" loading="lazy" class="wp-image-2297 size-full" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Testing-the-created-solution.jpg" alt="Testing the created solution from the Visual Studio Template" width="362" height="404" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Testing-the-created-solution.jpg 362w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/Testing-the-created-solution-269x300.jpg 269w" sizes="(max-width: 362px) 100vw, 362px" /></a>
  
  <p id="caption-attachment-2297" class="wp-caption-text">
    Testing the created solution
  </p>
</div>

## Add Docker Support to the Visual Studio Template

Adding Docker support to your Visual Studio Template is very simple. Right-click on your Template.Api project and select Add &#8211;> Docker Support. Select an operating system and click OK. This adds a Dockerfile and that&#8217;s it already.

[code language=&#8221;text&#8221;]  
FROM mcr.microsoft.com/dotnet/core/aspnet:3.1-buster-slim AS base  
WORKDIR /app  
EXPOSE 80

FROM mcr.microsoft.com/dotnet/core/sdk:3.1-buster AS build  
WORKDIR /src  
COPY ["Template.Api/Template.Api.csproj", "Template.Api/"]  
COPY ["Template.Service/Template.Service.csproj", "Template.Service/"]  
COPY ["Template.Data/Template.Data.csproj", "Template.Data/"]  
RUN dotnet restore "Template.Api/Template.Api.csproj"  
COPY . .  
WORKDIR "/src/Template.Api"  
RUN dotnet build "Template.Api.csproj" -c Release -o /app/build

FROM build AS publish  
RUN dotnet publish "Template.Api.csproj" -c Release -o /app/publish

FROM base AS final  
WORKDIR /app  
COPY &#8211;from=publish /app/publish .  
ENTRYPOINT ["dotnet", "Template.Api.dll"]  
[/code]

### Use Variables in the Dockerfile

Since we use variables everywhere, we have to use variables also in the Dockerfile because otherwise the projects wouldn&#8217;t be found since their name will change when the solution is created. Replace every Template with $ext_safeprojectname$ in the Dockerfile. When you are done export the project again. This time you only have to export the Api project.

After adding the new template to Visual Studio, create a new project and you will see the Dockerfile in your solution.

<div id="attachment_2298" style="width: 244px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-Dockerfile-is-in-the-created-project.jpg"><img aria-describedby="caption-attachment-2298" loading="lazy" class="wp-image-2298 size-full" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-Dockerfile-is-in-the-created-project.jpg" alt="The Dockerfile is in the created project from the Visual Studio Template" width="234" height="354" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-Dockerfile-is-in-the-created-project.jpg 234w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/07/The-Dockerfile-is-in-the-created-project-198x300.jpg 198w" sizes="(max-width: 234px) 100vw, 234px" /></a>
  
  <p id="caption-attachment-2298" class="wp-caption-text">
    The Dockerfile is in the created project
  </p>
</div>

## Conclusion

Use a Visual Studio Template to set up solutions that have the same structure. Microservices often have the same skeleton and are a great candidate for this. Templates allow you to install all Nuget packages and also to add Test projects. Another advantage is that even junior developers or interns can set up new projects.

You can find the code for the template without variables <a href="https://github.com/WolfgangOfner/VisualStudioTemplate/tree/withoutVariables" target="_blank" rel="noopener noreferrer">here</a> and the template with variables <a href="https://github.com/WolfgangOfner/VisualStudioTemplate" target="_blank" rel="noopener noreferrer">here</a>.