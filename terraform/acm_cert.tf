data "aws_route53_zone" "guydevops_zone" {
  name = "guydevops.com"
}

resource "aws_acm_certificate" "guydevops_cert" {
  domain_name       = "guydevops.com"
  validation_method = "DNS"

  tags = {
    Name = "guydevops.com"
  }
}

resource "aws_route53_record" "guydevops_cert_record" {
  for_each = {
    for dvo in aws_acm_certificate.guydevops_cert.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = data.aws_route53_zone.guydevops_zone.zone_id
}

resource "aws_acm_certificate_validation" "guydevops_cert_validation" {
  certificate_arn         = aws_acm_certificate.guydevops_cert.arn
  validation_record_fqdns = [for record in aws_route53_record.guydevops_cert_record : record.fqdn]
}
