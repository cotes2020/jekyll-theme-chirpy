
Set-Location -Path ../_site

echo "Searching HTML files"

$fileNames = Get-ChildItem -Path "$($PWD)\*.html" -Recurse | select -expand fullname

echo "$($fileNames.Length) files found"
echo "Replacing http://0.0.0.0 with https://www.programmingwithwolfgang.com in HTML files"

foreach ($filename in $filenames) 
{
 (  Get-Content $fileName) -replace "http://0.0.0.0","https://www.programmingwithwolfgang.com" | Set-Content $fileName
}

echo "HTML files replaced"

echo "Searching XML files"

$fileNames = Get-ChildItem -Path "$($PWD)\*.xml" -Recurse | select -expand fullname

echo "$($fileNames.Length) files found"
echo "Replacing http://0.0.0.0 with https://www.programmingwithwolfgang.com in HTML files"

foreach ($filename in $filenames) 
{
 (  Get-Content $fileName) -replace "http://0.0.0.0","https://www.programmingwithwolfgang.com" | Set-Content $fileName
}

echo "XML files replaced"

echo "Searching JSON files"

$fileNames = Get-ChildItem -Path "$($PWD)\*.JSON" -Recurse | select -expand fullname

echo "$($fileNames.Length) files found"
echo "Replacing http://0.0.0.0 with https://www.programmingwithwolfgang.com in HTML files"

foreach ($filename in $filenames) 
{
 (  Get-Content $fileName) -replace "http://0.0.0.0","https://www.programmingwithwolfgang.com" | Set-Content $fileName
}

echo "JSON files replaced"

echo "Searching TXT files"

$fileNames = Get-ChildItem -Path "$($PWD)\*.txt" -Recurse | select -expand fullname

echo "$($fileNames.Length) files found"
echo "Replacing http://0.0.0.0 with https://www.programmingwithwolfgang.com in HTML files"

foreach ($filename in $filenames) 
{
 (  Get-Content $fileName) -replace "http://0.0.0.0","https://www.programmingwithwolfgang.com" | Set-Content $fileName
}

echo "TXT files replaced"

