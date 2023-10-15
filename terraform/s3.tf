variable "secret_header" {
  type    = string
  default = "secret-header"
}

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
    condition {
      test     = "StringLike"
      variable = "aws:Referer"
      values   = ["${var.secret_header}"]
    }
  }
}

resource "null_resource" "remove_and_upload_to_s3" {
  provisioner "local-exec" {
    command = "aws s3 sync ~/repos/guydevops.com/_site s3://${aws_s3_bucket.guydevops-com_s3_bucket.id}"
  }
}