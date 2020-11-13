
Set-Location -Path ../_site

echo "Current path: $($PWD)"
echo "Searching HTML files"

$fileNames = Get-ChildItem -Path "$($PWD)\*.html" -Recurse | select -expand fullname

echo "$($fileNames.Length) files found"
echo "Replacing http://0.0.0.0:4000 with https://www.programmingwithwolfgang.com in HTML files"

foreach ($filename in $filenames) 
{
 (  Get-Content $fileName) -replace "http://0.0.0.0:4000","https://www.programmingwithwolfgang.com" | Set-Content $fileName -Force
}

echo "HTML files replaced"

echo "Searching XML files"

$fileNames = Get-ChildItem -Path "$($PWD)\*.xml" -Recurse | select -expand fullname

echo "$($fileNames.Length) files found"
echo "Replacing http://0.0.0.0:4000 with https://www.programmingwithwolfgang.com in HTML files"

foreach ($filename in $filenames) 
{
 (  Get-Content $fileName) -replace "http://0.0.0.0:4000","https://www.programmingwithwolfgang.com" | Set-Content $fileName -Force
}

echo "XML files replaced"

echo "Searching JSON files"

$fileNames = Get-ChildItem -Path "$($PWD)\*.JSON" -Recurse | select -expand fullname

echo "$($fileNames.Length) files found"
echo "Replacing http://0.0.0.0:4000 with https://www.programmingwithwolfgang.com in HTML files"

foreach ($filename in $filenames) 
{
 (  Get-Content $fileName) -replace "http://0.0.0.0:4000","https://www.programmingwithwolfgang.com" | Set-Content $fileName -Force
}

echo "JSON files replaced"

echo "Searching TXT files"

$fileNames = Get-ChildItem -Path "$($PWD)\*.txt" -Recurse | select -expand fullname

echo "$($fileNames.Length) files found"
echo "Replacing http://0.0.0.0:4000 with https://www.programmingwithwolfgang.com in HTML files"

foreach ($filename in $filenames) 
{
 (  Get-Content $fileName) -replace "http://0.0.0.0:4000","https://www.programmingwithwolfgang.com" | Set-Content $fileName -Force
}

echo "TXT files replaced"

Set-Location -Path ../

echo "Current path: $($PWD)"
echo "Starting to rename files from JPG to jpg"

Get-Childitem -Recurse | Where-Object {$_.Extension -cmatch "JPG"} | Rename-Item -NewName { $_.Name -replace '.JPG','.jpg' }

echo "Finished renaming files form JPG to jpg"