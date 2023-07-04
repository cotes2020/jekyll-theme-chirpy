resource "aws_s3_bucket" "guydevops-com_s3_bucket" {
  bucket = "guydevops.com"
  tags = {
    Name        = "guydevops"
    Environment = "prod"
  }
}


resource "aws_s3_bucket_website_configuration" "guydevops-com_s3_bucket_website" {
  bucket = aws_s3_bucket.guydevops-com_s3_bucket.id

  index_document {
    suffix = "index.html"
  }
  #   error_document {
  #     key = "error.html"
  #   }

  #   routing_rule {
  #     condition {
  #       key_prefix_equals = "docs/"
  #     }
  #     redirect {
  #       replace_key_prefix_with = "documents/"
  #     }
  #   }
}


resource "aws_s3_bucket_public_access_block" "allow_public_access_to_site_block" {
  bucket = aws_s3_bucket.guydevops-com_s3_bucket.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "allow_public_access_to_site" {
  bucket = aws_s3_bucket.guydevops-com_s3_bucket.id
  policy = data.aws_iam_policy_document.allow_public_access_to_site_policy.json
}


data "aws_iam_policy_document" "allow_public_access_to_site_policy" {

  statement {
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    actions = [
      "s3:GetObject"
    ]

    resources = [
      aws_s3_bucket.guydevops-com_s3_bucket.arn,
      "${aws_s3_bucket.guydevops-com_s3_bucket.arn}/*",
    ]
  }
}

resource "null_resource" "remove_and_upload_to_s3" {
  provisioner "local-exec" {
    command = "aws s3 sync ~/repos/guydevops.com/_site s3://${aws_s3_bucket.guydevops-com_s3_bucket.id}"
  }
}