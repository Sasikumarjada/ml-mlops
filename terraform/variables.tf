variable "region" {
  type    = string
  default = "ap-south-1"
}

variable "project" {
  type    = string
  default = "iris-ml"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "alert_email" {
  description = "Email for CloudWatch alarm notifications (leave empty to skip)"
  type        = string
  default     = ""
}
