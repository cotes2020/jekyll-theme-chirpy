#$fileNames = Get-ChildItem "\*.html" -Recurse | select -expand fullname
#
#foreach ($filename in $filenames) 
#{
#  (  Get-Content $fileName) -replace "https://www.programmingwithwolfgang.com","https://www.programmingwithwolfgang.com" | Set-Content $fileName
#}

Set-Location ../
#Set-LocalGroup _site

Get-Location
