locals {
  folder_id = var.yc_folder_id
}

# Account
resource "yandex_iam_service_account" "sa" {
  folder_id = local.folder_id
  name      = "tf-object-storage-sa"
}

resource "yandex_resourcemanager_folder_iam_member" "sa_storage_editor" {
  folder_id = local.folder_id
  role      = "storage.editor"
  member    = "serviceAccount:${yandex_iam_service_account.sa.id}"
}

# Keys
resource "yandex_iam_service_account_static_access_key" "sa_key" {
  service_account_id = yandex_iam_service_account.sa.id
  description        = "Access key for Object Storage"
}

# For unique names
resource "random_id" "suffix" {
  byte_length = 4
}

# Bucket
resource "yandex_storage_bucket" "bucket" {
  bucket        = "tf-bucket-${random_id.suffix.hex}"
  access_key    = yandex_iam_service_account_static_access_key.sa_key.access_key
  secret_key    = yandex_iam_service_account_static_access_key.sa_key.secret_key
  acl           = "private"
  force_destroy = true
}