# Update the table of contents in the built .docx and export it to PDF with Microsoft Word.
# Usage: powershell -File docs/thesis/export_pdf.ps1
$docx = Join-Path $PSScriptRoot "output\tugas-akhir-id.docx"
$pdf = Join-Path $PSScriptRoot "output\tugas-akhir-id.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Save()
    $doc.ExportAsFixedFormat([string]$pdf, 17)  # 17 = wdExportFormatPDF
    try { $doc.Close() } catch {}
    Write-Output "wrote $pdf"
} finally {
    $word.Quit()
}
