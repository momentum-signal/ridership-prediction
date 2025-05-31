import UserForm from "@/components/UserForm";

export default function Home() {
  return (    <div className="h-[100vh] flex flex-col overflow-hidden bg-gradient-to-br from-primary/10 to-background">
      <header className="w-full max-w-3xl mx-auto pt-1 sm:pt-2 flex flex-col items-center text-center px-2">
        <h1 className="uppercase text-xl xs:text-2xl sm:text-3xl md:text-4xl font-extrabold tracking-tight text-primary drop-shadow-lg break-words">
          Train Ridership Prediction
        </h1>
        <p className="text-xs xs:text-sm text-muted-foreground max-w-xs xs:max-w-md sm:max-w-2xl">
          Instantly predict train ridership between stations. Select your origin, destination, and date/time to get started.
        </p>
      </header>

      <main className="flex-1 flex items-center justify-center w-full px-2 sm:px-4 py-1 overflow-auto">
        <div className="w-full max-w-md bg-card/80 rounded-xl sm:rounded-2xl shadow-lg p-3 sm:p-5 border border-border backdrop-blur-md">
          <UserForm />
        </div>
      </main>

      <footer className="w-full max-w-3xl mx-auto py-1 text-center text-[10px] xs:text-xs text-muted-foreground px-2">
        &copy; {new Date().getFullYear()} Train Ridership Predictor. All rights reserved.
      </footer>
    </div>
  );
}
