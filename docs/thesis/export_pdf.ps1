# Update the table of contents in a built .docx and export it to PDF with Microsoft Word.
# Usage: powershell -File docs/thesis/export_pdf.ps1 [-Name tugas-akhir-id|proposal-ta-id]
param([string]$Name = "tugas-akhir-id")
$docx = Join-Path $PSScriptRoot "output\$Name.docx"
$pdf = Join-Path $PSScriptRoot "output\$Name.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx)
    foreach ($toc in $doc.TablesOfContents) {
        $toc.Update()
        # Single-spaced entries, so the list's closing line does not spill onto an empty page before BAB 1.
        $toc.Range.ParagraphFormat.LineSpacingRule = 0  # 0 = wdLineSpaceSingle
        $toc.Range.ParagraphFormat.SpaceAfter = 4
        $toc.UpdatePageNumbers()  # the shorter list moves the chapters up a page
    }
    $doc.Save()
    $doc.ExportAsFixedFormat([string]$pdf, 17)  # 17 = wdExportFormatPDF
    try { $doc.Close() } catch {}
    Write-Output "wrote $pdf"
} finally {
    $word.Quit()
}
