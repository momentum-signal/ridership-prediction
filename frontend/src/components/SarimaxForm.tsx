import React, { useState, ChangeEvent, FormEvent } from "react";

const SarimaxForm: React.FC = () => {
    const [formData, setFormData] = useState<{
        day_of_week: string;
        hour_of_day: string;
        is_weekend: boolean;
        is_holiday: boolean;
        month: string;
    }>({
        day_of_week: "",
        hour_of_day: "",
        is_weekend: false,
        is_holiday: false,
        month: "",
    });

    const [prediction, setPrediction] = useState<number | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
        const { name, value, type, checked } = e.target;
        setFormData({
            ...formData,
            [name]: type === "checkbox" ? checked : value,
        });
    };

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setError(null);
        setPrediction(null);

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(formData),
            });

            if (!response.ok) {
                throw new Error("Failed to fetch prediction");
            }

            const data = await response.json();
            setPrediction(data.prediction);
        } catch (err: any) {
            setError(err.message);
        }
    };

    return (
        <div className="max-w-md mx-auto p-4 bg-white shadow-md rounded-md">
            <h2 className="text-xl font-bold mb-4">SARIMAX Prediction</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label className="block text-sm font-medium">Day of Week</label>
                    <input
                        type="number"
                        name="day_of_week"
                        value={formData.day_of_week}
                        onChange={handleChange}
                        className="w-full border rounded p-2"
                        required
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium">Hour of Day</label>
                    <input
                        type="number"
                        name="hour_of_day"
                        value={formData.hour_of_day}
                        onChange={handleChange}
                        className="w-full border rounded p-2"
                        required
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium">Is Weekend</label>
                    <input
                        type="checkbox"
                        name="is_weekend"
                        checked={formData.is_weekend}
                        onChange={handleChange}
                        className="ml-2"
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium">Is Holiday</label>
                    <input
                        type="checkbox"
                        name="is_holiday"
                        checked={formData.is_holiday}
                        onChange={handleChange}
                        className="ml-2"
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium">Month</label>
                    <input
                        type="number"
                        name="month"
                        value={formData.month}
                        onChange={handleChange}
                        className="w-full border rounded p-2"
                        required
                    />
                </div>
                <button
                    type="submit"
                    className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
                >
                    Predict
                </button>
            </form>

            {prediction && (
                <div className="mt-4 p-4 bg-green-100 text-green-800 rounded">
                    <strong>Prediction:</strong> {prediction}
                </div>
            )}

            {error && (
                <div className="mt-4 p-4 bg-red-100 text-red-800 rounded">
                    <strong>Error:</strong> {error}
                </div>
            )}
        </div>
    );
};

export default SarimaxForm;
