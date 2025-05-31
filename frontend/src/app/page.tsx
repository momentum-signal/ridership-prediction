import UserForm from "@/components/UserForm";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-primary/10 to-background px-1 sm:px-4 py-4 sm:py-8">
      <header className="w-full max-w-3xl mx-auto mb-6 sm:mb-8 flex flex-col items-center text-center px-2">
        <h1 className="uppercase text-2xl xs:text-3xl sm:text-4xl md:text-5xl font-extrabold mb-2 tracking-tight text-primary drop-shadow-lg break-words">
          Train Ridership Prediction
        </h1>
        <p className="text-sm xs:text-base sm:text-lg md:text-xl text-muted-foreground max-w-xs xs:max-w-md sm:max-w-2xl mb-2 sm:mb-4">
          Instantly predict train ridership between stations. Select your origin, destination, and date/time to get started.
        </p>
      </header>
      <main className="flex-1 flex flex-col items-center justify-center w-full">
        <div className="w-full max-w-xs xs:max-w-sm sm:max-w-md md:max-w-xl bg-card/80 rounded-xl sm:rounded-2xl shadow-lg p-2 xs:p-4 sm:p-8 border border-border backdrop-blur-md">
          <UserForm />
        </div>
      </main>
      <footer className="w-full max-w-3xl mx-auto mt-6 sm:mt-8 text-center text-[10px] xs:text-xs text-muted-foreground pb-1 sm:pb-2 px-2">
        &copy; {new Date().getFullYear()} Train Ridership Predictor. All rights reserved.
      </footer>
    </div>
  );
}
