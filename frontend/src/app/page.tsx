"use client";
import UserForm from "@/components/UserForm";

export default function Home() {
  return (
    <div className="h-[100vh] flex flex-col overflow-hidden bg-gradient-to-br from-primary/10 to-background">
      <main className="flex-1 flex items-center justify-center w-full px-2 sm:px-4 py-1 overflow-auto">
        <div className="w-full max-w-lg bg-card rounded-xl sm:rounded-2xl shadow-lg p-3 sm:p-5 border border-border backdrop-blur-md">
          <header className="w-full max-w-3xl mx-auto pt-1 sm:pt-2 flex flex-col items-center text-center">
            <h1 className="uppercase text-xl xs:text-2xl sm:text-3xl md:text-3xl font-bold pb-5 tracking-tight text-primary drop-shadow-lg break-words">
              Train Ridership Prediction
            </h1>
          </header>
          <UserForm />
        </div>
      </main>

      <footer className="w-full max-w-3xl mx-auto py-1 text-center text-[10px] xs:text-xs text-muted-foreground px-2">
        &copy; {new Date().getFullYear()} Train Ridership Predictor. All rights
        reserved.
      </footer>
    </div>
  );
}
