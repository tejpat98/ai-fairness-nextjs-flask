"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Shield, Upload, FileText, Loader2, CheckCircle2 } from "lucide-react"
import { useRouter } from "next/navigation"

const DOMAINS = ["Business", "Health", "Finance", "Justice", "Hiring", "Governance", "Education"] as const

type Domain = (typeof DOMAINS)[number]

export default function CreateReportPage() {
  const router = useRouter()
  const [selectedDomain, setSelectedDomain] = useState<Domain | "">("")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [resultId, setResultId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    const file = event.dataTransfer.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  const startPolling = (result_id: string) => {
    console.log("[v0] Starting polling for result ID:", result_id)
    setIsAnalyzing(true)
    setProgress(20)

    pollingIntervalRef.current = setInterval(async () => {
      try {
        console.log("[v0] Polling status check for:", result_id)
        const response = await fetch(`/api/check-status/${result_id}`, {
          cache: 'no-store', // Ensures you always get the latest status
        })
        const data = await response.json()

        console.log("[v0] Status check response:", data)

        if (data.status === "finished") {
          console.log("[v0] Analysis complete, redirecting to report")
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
          }
          setProgress(100)
          // Redirect to report page
          setTimeout(() => {
            router.push(`/report/${selectedDomain.toLowerCase()}/${result_id}`)
          }, 500)
        } else {
          // Increment progress while analysing
          setProgress((prev) => Math.min(prev + 10, 90))
        }
      } catch (error) {
        console.error("[v0] Error polling status:", error)
      }
    }, 2000)
  }

  const handleStart = async () => {
    if (!selectedDomain || !selectedFile) {
      return
    }

    setIsUploading(true)
    setProgress(10)

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)

      console.log("[v0] Uploading file to:", `/api/${selectedDomain.toLowerCase()}/upload`)

      const response = await fetch(`/api/${selectedDomain.toLowerCase()}/upload`, {
        method: "POST",
        body: formData,
      })

      const data = await response.json()
      console.log("[v0] Upload response:", data)

      if (data.result_id) {
        setResultId(data.result_id)
        setIsUploading(false)
        startPolling(data.result_id)
      }
    } catch (error) {
      console.error("[v0] Error uploading file:", error)
      setIsUploading(false)
      setProgress(0)
    }
  }

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
    }
  }, [])

  const isFormValid = selectedDomain && selectedFile

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="h-6 w-6 text-accent" />
            <span className="font-semibold text-lg text-foreground">AI Fairness | FDK Toolkit</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 md:py-12 max-w-3xl">
        <div className="mb-8">
          <h1 className="text-3xl md:text-4xl font-bold mb-3 text-foreground">FDK Analysis Report</h1>
          <p className="text-muted-foreground text-lg">
            Select your domain and upload your dataset to begin detecting potential biases in your AI system.
          </p>
        </div>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Select Analysis Domain</CardTitle>
            <CardDescription>Choose the domain that best matches your use case</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <Label htmlFor="domain-select">Domain</Label>
              <Select value={selectedDomain} onValueChange={(value) => setSelectedDomain(value as Domain)}>
                <SelectTrigger id="domain-select" className="w-full">
                  <SelectValue placeholder="Select a domain..." />
                </SelectTrigger>
                <SelectContent>
                  {DOMAINS.map((domain) => (
                    <SelectItem key={domain} value={domain}>
                      {domain}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Upload Dataset</CardTitle>
            <CardDescription>Upload your dataset file for bias analysis (CSV)</CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-accent/50 transition-colors cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                onChange={handleFileChange}
                accept=".csv"
              />
              <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              {selectedFile ? (
                <div className="flex items-center justify-center gap-2 text-foreground">
                  <FileText className="h-5 w-5 text-accent" />
                  <span className="font-medium">{selectedFile.name}</span>
                  <Badge variant="secondary" className="ml-2">
                    {(selectedFile.size / 1024).toFixed(2)} KB
                  </Badge>
                </div>
              ) : (
                <>
                  <p className="text-muted-foreground mb-2">Drag and drop your file here, or click to browse</p>
                  <p className="text-sm text-muted-foreground">Supported formats: CSV</p>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        {(isUploading || isAnalyzing) && (
          <Card className="mb-6">
            <CardContent className="pt-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="h-5 w-5 animate-spin text-accent" />
                        <span className="font-medium">Analysing dataset for bias...</span>
                      </>
                    ) : (
                      <>
                        <Loader2 className="h-5 w-5 animate-spin text-accent" />
                        <span className="font-medium">Uploading dataset...</span>
                      </>
                    )}
                  </div>
                  <span className="text-sm text-muted-foreground">{progress}%</span>
                </div>
                <Progress value={progress} className="w-full" />
                {resultId && (
                  <p className="text-sm text-muted-foreground">
                    Analysis ID: <span className="font-mono text-foreground">{resultId}</span>
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        <Button
          size="lg"
          className="w-full"
          disabled={!isFormValid || isUploading || isAnalyzing}
          onClick={handleStart}
        >
          {isUploading || isAnalyzing ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <CheckCircle2 className="mr-2 h-5 w-5" />
              Start Analysis
            </>
          )}
        </Button>
      </main>
    </div>
  )
}
