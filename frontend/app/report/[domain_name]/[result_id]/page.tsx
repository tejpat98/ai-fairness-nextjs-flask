import { notFound } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Shield, ArrowLeft, FileText, Download, AlertTriangle, CheckCircle2 } from "lucide-react"
import Link from "next/link"

interface ReportData {
  [key: string]: any
}

export default async function ReportPage({
  params,
}: {
  params: Promise<{ domain_name: string; result_id: string }>
}) {
  // Await the dynamic parameters from the URL
  const { domain_name, result_id } = await params

  // Fetch the report from the Flask API
  const res = await fetch(`http://backend:5000/api/report/${domain_name}/${result_id}`, {
    cache: "no-store",
  })

  if (!res.ok) {
    if (res.status === 404) notFound()
    return (
      <div className="min-h-screen bg-background">
        <header className="border-b border-border">
          <div className="container mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="h-6 w-6 text-accent" />
              <span className="font-semibold text-lg text-foreground">AI Fairness | FDK Toolkit</span>
            </div>
          </div>
        </header>
        <div className="container mx-auto px-4 py-16 text-center">
          <AlertTriangle className="h-16 w-16 text-destructive mx-auto mb-4" />
          <h1 className="text-2xl font-bold mb-2">Error Loading Report</h1>
          <p className="text-muted-foreground mb-6">Please try again later.</p>
          <Button asChild>
            <Link href="/">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Home
            </Link>
          </Button>
        </div>
      </div>
    )
  }

  const reports: ReportData = await res.json()

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="h-6 w-6 text-accent" />
            <span className="font-semibold text-lg text-foreground">AI Fairness | FDK Toolkit</span>
          </div>
          <Button variant="outline" size="sm" asChild>
            <Link href="/">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Link>
          </Button>
        </div>
      </header>

      {/* Report Content */}
      <main className="container mx-auto px-4 py-8 md:py-12">
        <div className="max-w-5xl mx-auto">
          {/* Report Header */}
          <div className="mb-8">
            <div className="flex items-center gap-2 mb-4">
              <Button variant="ghost" size="sm" asChild>
                <Link href="/create-report">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  New Analysis
                </Link>
              </Button>
            </div>

            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-4">
              <div>
                <h1 className="text-3xl md:text-4xl font-bold mb-2 text-foreground capitalize">
                  {domain_name.replace(/-/g, " ")} Analysis
                </h1>
                <p className="text-muted-foreground">Report ID: {result_id}</p>
              </div>
              <Button variant="outline" className="gap-2 bg-transparent">
                <Download className="h-4 w-4" />
                Download
              </Button>
            </div>

            <Badge variant="secondary" className="mb-2">
              <CheckCircle2 className="h-3 w-3 mr-1" />
              Analysis Complete
            </Badge>
          </div>

          <Separator className="mb-8" />

          {/* Report Sections */}
          <div className="space-y-6">
            {/* Grid container for Audit Report and Audit Summary on desktop */}
            <div className="lg:grid lg:grid-cols-2 lg:gap-6 space-y-6 lg:space-y-0">
              {Object.entries(reports)
                .filter(([filename]) => {
                  const displayName = filename
                    .replace(result_id + "_", "")
                    .replace(/_/g, " ")
                    .toLowerCase()
                  return displayName.includes("audit report") || displayName.includes("audit summary")
                })
                .map(([filename, content]) => {
                  const displayName = filename
                    .replace(result_id + "_", "")
                    .replace(/_/g, " ")
                    .split(" ")
                    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(" ")

                  return (
                    <Card key={filename} className="p-6 bg-card border-border">
                      <div className="flex items-start gap-3 mb-4">
                        <div className="h-10 w-10 rounded-lg bg-accent/10 flex items-center justify-center flex-shrink-0">
                          <FileText className="h-5 w-5 text-accent" />
                        </div>
                        <div className="flex-1">
                          <h2 className="text-xl md:text-2xl font-semibold text-foreground">{displayName}</h2>
                        </div>
                      </div>

                      <Separator className="mb-4" />

                      <div className="bg-muted/30 p-4 rounded-lg overflow-auto max-h-[600px]">
                        {typeof content === "object" ? (
                          <pre className="text-sm text-foreground font-mono leading-relaxed">
                            {JSON.stringify(content, null, 2)}
                          </pre>
                        ) : (
                          <p className="whitespace-pre-wrap text-sm text-foreground leading-relaxed">{content}</p>
                        )}
                      </div>
                    </Card>
                  )
                })}
            </div>

            {/* Other report sections remain stacked */}
            {Object.entries(reports)
              .filter(([filename]) => {
                const displayName = filename
                  .replace(result_id + "_", "")
                  .replace(/_/g, " ")
                  .toLowerCase()
                return !displayName.includes("audit report") && !displayName.includes("audit summary")
              })
              .map(([filename, content]) => {
                const displayName = filename
                  .replace(result_id + "_", "")
                  .replace(/_/g, " ")
                  .split(" ")
                  .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                  .join(" ")

                return (
                  <Card key={filename} className="p-6 bg-card border-border">
                    <div className="flex items-start gap-3 mb-4">
                      <div className="h-10 w-10 rounded-lg bg-accent/10 flex items-center justify-center flex-shrink-0">
                        <FileText className="h-5 w-5 text-accent" />
                      </div>
                      <div className="flex-1">
                        <h2 className="text-xl md:text-2xl font-semibold text-foreground">{displayName}</h2>
                      </div>
                    </div>

                    <Separator className="mb-4" />

                    <div className="bg-muted/30 p-4 rounded-lg overflow-auto max-h-[600px]">
                      {typeof content === "object" ? (
                        <pre className="text-sm text-foreground font-mono leading-relaxed">
                          {JSON.stringify(content, null, 2)}
                        </pre>
                      ) : (
                        <p className="whitespace-pre-wrap text-sm text-foreground leading-relaxed">{content}</p>
                      )}
                    </div>
                  </Card>
                )
              })}
          </div>

          {/* Actions */}
          <div className="mt-8 flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" asChild>
              <Link href="/create-report">Run New Analysis</Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <Link href="/">Back to Home</Link>
            </Button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center">
            <p className="text-sm text-muted-foreground">2025 AI Fairness | FDK Toolkit ™</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
