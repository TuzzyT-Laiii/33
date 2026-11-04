param(
  [Parameter(Mandatory=$true)][string]$InputDocx,
  [Parameter(Mandatory=$true)][string]$OutputPdf,
  [Parameter(Mandatory=$true)][string]$AuditJson
)
$ErrorActionPreference = 'Stop'
$inputPath = (Resolve-Path -LiteralPath $InputDocx).Path
$pdfPath = [System.IO.Path]::GetFullPath($OutputPdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $document = $word.Documents.Open($inputPath, $false, $true)
  try {
    $document.Repaginate()
    $document.ExportAsFixedFormat($pdfPath, 17)
    $tables = @()
    for ($index = 1; $index -le $document.Tables.Count; $index++) {
      $range = $document.Tables.Item($index).Range
      $startRange = $document.Range($range.Start, $range.Start)
      $endRange = $document.Range($range.End - 1, $range.End - 1)
      $firstPage = $startRange.Information(3)
      $lastPage = $endRange.Information(3)
      if ($firstPage -ne $lastPage) { throw "表格 $index 跨页：$firstPage 至 $lastPage" }
      $tables += @{index=$index; start_page=$firstPage; end_page=$lastPage; rows=$document.Tables.Item($index).Rows.Count}
    }
    @{pages=$document.ComputeStatistics(2); tables=$tables; exporter='Microsoft Word native PDF export'; docx_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $inputPath).Hash; pdf_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $pdfPath).Hash} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $AuditJson -Encoding UTF8
  } finally {
    $document.Close(0)
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null
  }
} finally {
  $word.Quit()
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
