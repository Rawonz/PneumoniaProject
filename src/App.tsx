import { useState, useCallback, useRef } from "react"
import { Upload, X, Activity, AlertCircle, CheckCircle2, Loader2, FlaskConical } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { cn } from "@/lib/utils"

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000"

type PredictionResult = {
  prediction: "NORMAL" | "PNEUMONIA"
  confidence: number
  probabilities: { NORMAL: number; PNEUMONIA: number }
  inference_ms: number
}

type UploadState = "idle" | "dragging" | "uploading" | "done" | "error"

export function App() {
  const [uploadState, setUploadState] = useState<UploadState>("idle")
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [errorMsg, setErrorMsg] = useState<string>("")
  const inputRef = useRef<HTMLInputElement>(null)

  const reset = useCallback(() => {
    setUploadState("idle")
    setFile(null)
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setResult(null)
    setErrorMsg("")
  }, [preview])

  const handleFile = useCallback(async (f: File) => {
    if (!f.type.startsWith("image/")) {
      setErrorMsg("Please upload a JPEG or PNG image.")
      setUploadState("error")
      return
    }
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setErrorMsg("")
    setUploadState("uploading")

    const form = new FormData()
    form.append("file", f)

    try {
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail ?? "Prediction failed")
      }
      const data: PredictionResult = await res.json()
      setResult(data)
      setUploadState("done")
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error"
      setErrorMsg(msg)
      setUploadState("error")
    }
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setUploadState("idle")
      const f = e.dataTransfer.files[0]
      if (f) handleFile(f)
    },
    [handleFile]
  )

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setUploadState("dragging")
  }
  const onDragLeave = () => setUploadState("idle")

  const isPneumonia = result?.prediction === "PNEUMONIA"
  const confidencePct = result ? Math.round(result.confidence * 100) : 0

  return (
    <div className="min-h-svh bg-background text-foreground">
      {/* ── Header ── */}
      <header className="border-b bg-card">
        <div className="mx-auto flex max-w-4xl items-center gap-3 px-6 py-4">
          <div className="flex size-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <FlaskConical className="size-5" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">PneumoScan</h1>
            <p className="text-xs text-muted-foreground">AI-Powered Chest X-Ray Analysis</p>
          </div>
          <Badge variant="outline" className="ml-auto text-xs">
            VGG19 · Experimental
          </Badge>
        </div>
      </header>

      <main className="mx-auto max-w-4xl space-y-8 px-6 py-10">
        {/* ── Disclaimer ── */}
        <Alert>
          <AlertCircle className="size-4" />
          <AlertTitle>For research and educational purposes only</AlertTitle>
          <AlertDescription className="text-sm">
            This tool is not a certified medical device. Results should not replace professional
            medical diagnosis. Always consult a qualified radiologist or physician.
          </AlertDescription>
        </Alert>

        {/* ── Main content ── */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* ── Upload panel ── */}
          <div className="flex flex-col gap-4">
            <div>
              <h2 className="text-base font-semibold">Upload X-Ray Image</h2>
              <p className="text-sm text-muted-foreground">
                Drag &amp; drop or click to select a chest X-ray (JPEG / PNG, max 10 MB)
              </p>
            </div>

            {/* Drop zone */}
            <div
              role="button"
              tabIndex={0}
              aria-label="Upload X-ray image"
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
              className={cn(
                "relative flex min-h-52 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 transition-colors focus-visible:ring-2 focus-visible:ring-ring",
                uploadState === "dragging"
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/60 hover:bg-muted/40",
                (uploadState === "uploading" || uploadState === "done") && "pointer-events-none"
              )}
            >
              {uploadState === "uploading" ? (
                <Loader2 className="size-10 animate-spin text-primary" />
              ) : preview ? (
                <img
                  src={preview}
                  alt="X-ray preview"
                  className="max-h-40 max-w-full rounded-md object-contain"
                />
              ) : (
                <>
                  <div className="flex size-14 items-center justify-center rounded-full bg-muted">
                    <Upload className="size-6 text-muted-foreground" />
                  </div>
                  <p className="text-sm font-medium">
                    {uploadState === "dragging" ? "Drop to upload" : "Click or drag image here"}
                  </p>
                  <p className="text-xs text-muted-foreground">JPEG, PNG — up to 10 MB</p>
                </>
              )}
              <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png"
                className="sr-only"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) handleFile(f)
                  e.target.value = ""
                }}
              />
            </div>

            {/* File info + reset */}
            {file && (
              <div className="flex items-center justify-between rounded-lg border bg-muted/30 px-4 py-2.5 text-sm">
                <span className="truncate text-muted-foreground">
                  {file.name} &nbsp;·&nbsp; {(file.size / 1024).toFixed(0)} KB
                </span>
                <Button variant="ghost" size="icon-sm" onClick={reset} aria-label="Remove file">
                  <X className="size-4" />
                </Button>
              </div>
            )}

            {/* Error */}
            {uploadState === "error" && (
              <Alert variant="destructive">
                <AlertCircle className="size-4" />
                <AlertTitle>Analysis failed</AlertTitle>
                <AlertDescription>{errorMsg}</AlertDescription>
              </Alert>
            )}
          </div>

          {/* ── Results panel ── */}
          <div className="flex flex-col gap-4">
            <div>
              <h2 className="text-base font-semibold">Analysis Result</h2>
              <p className="text-sm text-muted-foreground">
                AI prediction will appear here after upload
              </p>
            </div>

            {/* Placeholder */}
            {!result && uploadState !== "uploading" && (
              <div className="flex min-h-52 flex-col items-center justify-center gap-3 rounded-xl border border-dashed bg-muted/20 p-8 text-center">
                <Activity className="size-10 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground">
                  Upload an X-ray to see the prediction
                </p>
              </div>
            )}

            {/* Analyzing skeleton */}
            {uploadState === "uploading" && (
              <div className="flex min-h-52 flex-col items-center justify-center gap-4 rounded-xl border bg-muted/10 p-8">
                <Loader2 className="size-8 animate-spin text-primary" />
                <p className="text-sm font-medium">Analyzing image…</p>
                <p className="text-xs text-muted-foreground">This usually takes a few seconds</p>
              </div>
            )}

            {/* Result card */}
            {result && (
              <div
                className={cn(
                  "flex flex-col gap-5 rounded-xl border-2 p-6 transition-colors",
                  isPneumonia
                    ? "border-destructive/40 bg-destructive/5"
                    : "border-green-500/40 bg-green-500/5"
                )}
              >
                {/* Verdict */}
                <div className="flex items-center gap-3">
                  <div
                    className={cn(
                      "flex size-12 items-center justify-center rounded-full",
                      isPneumonia ? "bg-destructive/10" : "bg-green-500/10"
                    )}
                  >
                    {isPneumonia ? (
                      <AlertCircle className="size-6 text-destructive" />
                    ) : (
                      <CheckCircle2 className="size-6 text-green-600 dark:text-green-400" />
                    )}
                  </div>
                  <div>
                    <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                      Prediction
                    </p>
                    <p
                      className={cn(
                        "text-2xl font-bold tracking-tight",
                        isPneumonia ? "text-destructive" : "text-green-700 dark:text-green-400"
                      )}
                    >
                      {result.prediction}
                    </p>
                  </div>
                </div>

                {/* Confidence bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-medium">
                    <span>Confidence</span>
                    <span>{confidencePct}%</span>
                  </div>
                  <Progress
                    value={confidencePct}
                    className={cn(
                      "h-2",
                      isPneumonia
                        ? "[&>[data-slot=progress-indicator]]:bg-destructive"
                        : "[&>[data-slot=progress-indicator]]:bg-green-500"
                    )}
                  />
                </div>

                {/* Probability breakdown */}
                <div className="space-y-2">
                  <p className="text-xs font-medium text-muted-foreground">Class Probabilities</p>
                  <div className="grid grid-cols-2 gap-3">
                    {(["NORMAL", "PNEUMONIA"] as const).map((cls) => {
                      const pct = Math.round(result.probabilities[cls] * 100)
                      const isActive = result.prediction === cls
                      return (
                        <div
                          key={cls}
                          className={cn(
                            "rounded-lg border p-3 text-center transition-colors",
                            isActive
                              ? cls === "PNEUMONIA"
                                ? "border-destructive/30 bg-destructive/5"
                                : "border-green-500/30 bg-green-500/5"
                              : "border-border bg-muted/20"
                          )}
                        >
                          <p className="text-xs text-muted-foreground">{cls}</p>
                          <p
                            className={cn(
                              "text-xl font-bold tabular-nums",
                              isActive
                                ? cls === "PNEUMONIA"
                                  ? "text-destructive"
                                  : "text-green-700 dark:text-green-400"
                                : "text-muted-foreground"
                            )}
                          >
                            {pct}%
                          </p>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Inference time */}
                <p className="text-xs text-muted-foreground">
                  Inference time: {result.inference_ms} ms
                </p>

                {/* Analyse another */}
                <Button variant="outline" size="sm" onClick={reset} className="w-full">
                  Analyse another image
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* ── How it works ── */}
        <section className="rounded-xl border bg-card p-6">
          <h2 className="mb-4 text-sm font-semibold">How it works</h2>
          <div className="grid gap-4 text-sm sm:grid-cols-3">
            {[
              {
                step: "1",
                title: "Upload X-Ray",
                body: "Select or drag in a frontal chest X-ray (PA or AP view) in JPEG or PNG format.",
              },
              {
                step: "2",
                title: "VGG19 Inference",
                body: "The image is pre-processed and fed through a VGG19 model trained on 5,800 labelled scans.",
              },
              {
                step: "3",
                title: "Review Result",
                body: "The predicted class (Normal or Pneumonia) is displayed with a confidence score.",
              },
            ].map(({ step, title, body }) => (
              <div key={step} className="flex gap-3">
                <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">
                  {step}
                </span>
                <div>
                  <p className="font-medium">{title}</p>
                  <p className="mt-0.5 text-muted-foreground">{body}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>

      {/* ── Footer ── */}
      <footer className="border-t py-6 text-center text-xs text-muted-foreground">
        PneumoScan · Research Prototype · Not for clinical use · Model trained on
        Kaggle Chest X-Ray Images (Pneumonia) dataset
      </footer>
    </div>
  )
}

export default App
