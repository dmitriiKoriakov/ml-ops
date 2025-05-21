variable "yc_iam_token" {
  description = "IAM токен YCloud"
  type        = string
  sensitive   = true
}

variable "yc_cloud_id" {
  description = "ID облака"
  type        = string
}

variable "yc_folder_id" {
  description = "ID каталога (folder)"
  type        = string
}