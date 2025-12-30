"use client"

import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { CheckCircle2, Shield, BarChart3, AlertTriangle, Search, FileText, ArrowRight, Menu } from "lucide-react"
import Link from 'next/link'

export default function Page() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="h-6 w-6 text-accent" />
            <span className="font-semibold text-lg text-foreground">AI Fairness | FDK Toolkit</span>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <a href="#features" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Features
            </a>
            <a href="https://github.com/AI-Fairness-com/FDK-Toolkit" className="text-sm text-muted-foreground hover:text-foreground transition-colors" target="_blank">
              Github
            </a>
            <a href="/docs" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Docs
            </a>
            <a href="/about" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              About
            </a>
            <a href="/contact" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Contact
            </a>
          </nav>
          <Button variant="outline" size="sm" className="hidden md:flex bg-transparent">
            <Link href="/create-report">Create Report</Link>
          </Button>
          <Sheet>
            <SheetTrigger asChild className="md:hidden">
              <Button variant="ghost" size="icon">
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-[280px]">
              <nav className="flex flex-col gap-6 mt-8">
                <a href="#features" className="text-base text-foreground hover:text-accent transition-colors">
                  Features
                </a>
                <a href="https://github.com/AI-Fairness-com/FDK-Toolkit" className="text-base text-foreground hover:text-accent transition-colors">
                  Github
                </a>
                <a href="#/docs" className="text-base text-foreground hover:text-accent transition-colors">
                  Docs
                </a>
                <a href="/about" className="text-base text-foreground hover:text-accent transition-colors">
                  About
                </a>
                <a href="/contact" className="text-base text-foreground hover:text-accent transition-colors">
                  Contact
                </a>
                <Separator />
                <Button className="w-full">
                  <Link href="/create-report">Create Report</Link>
                </Button>
              </nav>
            </SheetContent>
          </Sheet>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16 md:py-24 lg:py-32">
        <div className="max-w-4xl mx-auto text-center">
          <Badge variant="secondary" className="mb-4 px-3 py-1">
            Built for AI Safety
          </Badge>
          <h1 className="text-4xl md:text-5xl lg:text-7xl font-bold mb-6 text-balance leading-tight">
            Detect bias in your AI before it becomes a problem
          </h1>
          <p className="text-base md:text-lg lg:text-xl text-muted-foreground mb-8 text-pretty max-w-2xl mx-auto leading-relaxed">
            Analyse datasets from your AI processes to identify fairness issues, demographic disparities, and hidden
            biases—ensuring your models serve everyone equally.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90">
              <Link href="/create-report">Start Free Analysis</Link>
            </Button>
            <Button size="lg" variant="outline">
              <Link href="/examples">Example Use Case</Link>
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>

          <Separator className="my-12 md:my-16 max-w-2xl mx-auto" />
          <div>
            <p className="text-sm text-muted-foreground mb-6">Trusted by AI teams at</p>
            <div className="flex flex-wrap items-center justify-center gap-6 md:gap-8 lg:gap-12 opacity-60">
              <span className="text-base md:text-lg font-semibold">TechCorp</span>
              <span className="text-base md:text-lg font-semibold">DataLabs</span>
              <span className="text-base md:text-lg font-semibold">MLVentures</span>
              <span className="text-base md:text-lg font-semibold">AIInnovate</span>
              <span className="text-base md:text-lg font-semibold">NeuralWorks</span>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="container mx-auto px-4 py-16 md:py-20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12 md:mb-16">
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-4 text-balance">
              Comprehensive Bias Detection
            </h2>
            <p className="text-base md:text-lg text-muted-foreground text-pretty max-w-2xl mx-auto">
              Our toolkit analyses every aspect of your AI pipeline to surface hidden biases and fairness concerns.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
            <Card className="p-6 bg-card border-border hover:border-accent/50 transition-colors">
              <div className="h-12 w-12 rounded-lg bg-accent/10 flex items-center justify-center mb-4">
                <Search className="h-6 w-6 text-accent" />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-card-foreground">Automated Detection</h3>
              <p className="text-muted-foreground leading-relaxed">
                Automatically scan your datasets for statistical disparities across protected attributes like gender,
                race, and age.
              </p>
            </Card>

            <Card className="p-6 bg-card border-border hover:border-accent/50 transition-colors">
              <div className="h-12 w-12 rounded-lg bg-accent/10 flex items-center justify-center mb-4">
                <Shield className="h-6 w-6 text-accent" />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-card-foreground">Privacy First</h3>
              <p className="text-muted-foreground leading-relaxed">
                As we are open-source. All analysis can happen in your environment. Your sensitive data never leaves your infrastructure.
              </p>
            </Card>

            <Card className="p-6 bg-card border-border hover:border-accent/50 transition-colors">
              <div className="h-12 w-12 rounded-lg bg-accent/10 flex items-center justify-center mb-4">
                <CheckCircle2 className="h-6 w-6 text-accent" />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-card-foreground">CI/CD Integration</h3>
              <p className="text-muted-foreground leading-relaxed">
                Integrate our open-source toolkit bias checks directly into your deployment pipeline to catch issues before they reach production.
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section id="how-it-works" className="container mx-auto px-4 py-16 md:py-20">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12 md:mb-16">
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-4 text-balance">How It Works</h2>
            <p className="text-base md:text-lg text-muted-foreground text-pretty">
              Four simple steps to uncover and address bias in your AI systems
            </p>
          </div>

          <div className="space-y-4 md:space-y-6">
            <Card className="p-4 md:p-6">
              <div className="flex gap-4 md:gap-6">
                <div className="flex-shrink-0">
                  <div className="h-10 w-10 md:h-12 md:w-12 rounded-full bg-accent text-accent-foreground flex items-center justify-center font-bold text-base md:text-lg">
                    1
                  </div>
                </div>
                <div className="pt-1">
                  <h3 className="text-lg md:text-xl font-semibold mb-2 text-foreground">Connect Your Data</h3>
                  <p className="text-sm md:text-base text-muted-foreground leading-relaxed">
                    Upload your dataset or connect to your existing data warehouse. We currently support CSV datasets.
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-4 md:p-6">
              <div className="flex gap-4 md:gap-6">
                <div className="flex-shrink-0">
                  <div className="h-10 w-10 md:h-12 md:w-12 rounded-full bg-accent text-accent-foreground flex items-center justify-center font-bold text-base md:text-lg">
                    2
                  </div>
                </div>
                <div className="pt-1">
                  <h3 className="text-lg md:text-xl font-semibold mb-2 text-foreground">Run Analysis</h3>
                  <p className="text-sm md:text-base text-muted-foreground leading-relaxed">
                    Our engine identifies protected attributes and then applies statistical fairness metrics, disparity tests, and demographic parity checks
                    across your dataset.
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-4 md:p-6">
              <div className="flex gap-4 md:gap-6">
                <div className="flex-shrink-0">
                  <div className="h-10 w-10 md:h-12 md:w-12 rounded-full bg-accent text-accent-foreground flex items-center justify-center font-bold text-base md:text-lg">
                    3
                  </div>
                </div>
                <div className="pt-1">
                  <h3 className="text-lg md:text-xl font-semibold mb-2 text-foreground">Get Actionable Insights</h3>
                  <p className="text-sm md:text-base text-muted-foreground leading-relaxed">
                    Review detailed reports with bias scores, visualisations, and specific recommendations to improve
                    your model's fairness.
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4 py-16 md:py-20">
        <div className="max-w-4xl mx-auto">
          <Card className="p-6 md:p-8 lg:p-12 bg-accent/5 border-accent/20">
            <div className="text-center">
              <h2 className="text-2xl md:text-3xl lg:text-4xl font-bold mb-4 text-balance text-foreground">
                Build AI systems everyone can trust
              </h2>
              <p className="text-base md:text-lg text-muted-foreground mb-8 text-pretty max-w-2xl mx-auto">
                Start detecting and fixing bias in your AI models today. Use out online toolkit to analyse a dataset today.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90">
                  <Link href="/create-report">Start Free Analysis</Link>
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </section>

      {/* Footer */}
 <footer className="border-t border-border mt-16 md:mt-20">
        <div className="container mx-auto px-4 py-8 md:py-12">
          <div className="max-w-4xl mx-auto">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8 text-center">
              <div>
                <h4 className="font-semibold mb-4 text-foreground">Product</h4>
                <ul className="space-y-2">
                  <li>
                    <a href="#features" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      Features
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      Pricing
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      API
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-4 text-foreground">Resources</h4>
                <ul className="space-y-2">
                  <li>
                    <a href="/docs" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      Documentation
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      Guides
                    </a>
                  </li>
                  <li>
                    <a href="/blog" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      Blog
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-4 text-foreground">Company</h4>
                <ul className="space-y-2">
                  <li>
                    <a href="/about" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      About
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      Careers
                    </a>
                  </li>
                  <li>
                    <a href="/contact" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      Contact
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-4 text-foreground">Legal</h4>
                <ul className="space-y-2">
                  <li>
                    <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      Privacy
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      Terms
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                      Security
                    </a>
                  </li>
                </ul>
              </div>
            </div>
            <Separator className="my-6 md:my-8" />
            <div className="text-center">
              <p className="text-sm text-muted-foreground">2025 AI Fairness | FDK Toolkit ™</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
