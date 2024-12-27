'use client';

import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const SweepDashboard = () => {
  const [sweeps, setSweeps] = useState([]);
  const [selectedSweep, setSelectedSweep] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSweeps = async () => {
      try {
        const response = await fetch('/api/sweeps');
        const data = await response.json();
        setSweeps(data);
      } catch (error) {
        console.error('Error fetching sweeps:', error);
      }
    };
    fetchSweeps();
  }, []);

  useEffect(() => {
    const fetchResults = async () => {
      if (!selectedSweep) return;
      
      try {
        setLoading(true);
        const response = await fetch(`/api/sweeps/${selectedSweep.id}/results`);
        const data = await response.json();
        // Sort by created_at and add index for x-axis
        const sortedData = data
          .sort((a, b) => a.created_at - b.created_at)
          .map((result, index) => ({
            ...result,
            index: index + 1
          }));
        setResults(sortedData);
      } catch (error) {
        console.error('Error fetching results:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchResults();
  }, [selectedSweep]);

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className="w-64 bg-gray-100 p-4 border-r">
        <h2 className="text-xl font-bold mb-4">Sweeps</h2>
        <div className="space-y-2">
          {sweeps.map((sweep) => (
            <div
              key={sweep.id}
              className={`p-2 rounded cursor-pointer ${
                selectedSweep?.id === sweep.id
                  ? 'bg-blue-500 text-white'
                  : 'hover:bg-gray-200'
              }`}
              onClick={() => setSelectedSweep(sweep)}
            >
              {sweep.name}
            </div>
          ))}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 p-4 overflow-auto">
        {selectedSweep ? (
          <>
            <h1 className="text-2xl font-bold mb-4">
              Sweep: {selectedSweep.name}
            </h1>
            
            {/* Results Chart */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle>Optimization Progress</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={results}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="index"
                        label={{ value: 'Experiment Number', position: 'bottom' }}
                      />
                      <YAxis
                        label={{
                          value: 'Value',
                          angle: -90,
                          position: 'insideLeft'
                        }}
                      />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="#8884d8"
                        dot={{ r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Results Table */}
            <Card>
              <CardHeader>
                <CardTitle>Experiments</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="p-2 text-left">#</th>
                        <th className="p-2 text-left">Parameters</th>
                        <th className="p-2 text-left">Value</th>
                        <th className="p-2 text-left">Status</th>
                        <th className="p-2 text-left">Created</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((result, index) => (
                        <tr key={index} className="border-b">
                          <td className="p-2">{index + 1}</td>
                          <td className="p-2">
                            {Object.entries(result.parameters)
                              .map(([key, value]) => `${key}: ${value.toFixed(4)}`)
                              .join(', ')}
                          </td>
                          <td className="p-2">
                            {result.value !== null ? result.value.toFixed(4) : 'N/A'}
                          </td>
                          <td className="p-2">{result.status}</td>
                          <td className="p-2">
                            {new Date(result.created_at * 1000).toLocaleString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </>
        ) : (
          <div className="text-center text-gray-500 mt-10">
            Select a sweep from the sidebar to view results
          </div>
        )}
      </div>
    </div>
  );
};

export default SweepDashboard;