import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2, Star, StarHalf, Search } from 'lucide-react';

const MovieRecommender = () => {
  const [userId, setUserId] = useState('');
  const [selectedUser, setSelectedUser] = useState(null);
  const [userHistory, setUserHistory] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!userId.trim()) return;

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/recommendations/${userId}`);
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Felhasználó nem található');
        }
        throw new Error('Hiba történt az ajánlások betöltésekor');
      }
      const data = await response.json();

      setSelectedUser(userId);
      setUserHistory(data.userHistory);
      setRecommendations(data.recommendations);
    } catch (error) {
      setError('Hiba történt a felhasználó betöltésekor. Kérjük, próbálja újra.');
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const RatingStars = ({ rating }) => {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    
    return (
      <div className="flex items-center gap-1">
        {[...Array(fullStars)].map((_, i) => (
          <Star key={i} className="w-4 h-4 fill-yellow-400 text-yellow-400" />
        ))}
        {hasHalfStar && <StarHalf className="w-4 h-4 fill-yellow-400 text-yellow-400" />}
        <span className="ml-2 text-sm text-gray-600">({rating.toFixed(1)})</span>
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-4 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>MovieLens Film Ajánló</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <form onSubmit={handleSearch} className="flex gap-2">
              <div className="flex-1">
                <input
                  type="number"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="Adja meg a felhasználó ID-t..."
                  className="w-full px-4 py-2 border rounded-md"
                  min="1"
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300 flex items-center gap-2"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                Keresés
              </button>
            </form>

            {error && (
              <div className="p-4 bg-red-50 text-red-600 rounded-md">
                {error}
              </div>
            )}

            {loading && (
              <div className="flex justify-center items-center py-8">
                <Loader2 className="w-8 h-8 animate-spin" />
              </div>
            )}

            {selectedUser && !loading && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Korábbi értékelések</h3>
                  <div className="grid gap-3">
                    {userHistory.map(movie => (
                      <div
                        key={movie.id}
                        className="p-3 bg-gray-50 rounded-lg"
                      >
                        <div className="flex justify-between items-start">
                          <div>
                            <h4 className="font-medium">{movie.title}</h4>
                            <div className="text-sm text-gray-600">
                              {movie.genres.join(', ')}
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              Értékelve: {new Date(movie.timestamp).toLocaleDateString()}
                            </div>
                          </div>
                          <RatingStars rating={movie.rating} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">Ajánlott filmek</h3>
                  <div className="grid gap-3">
                    {recommendations.map(movie => (
                      <div
                        key={movie.id}
                        className="p-3 bg-blue-50 rounded-lg"
                      >
                        <div className="flex justify-between items-start">
                          <div>
                            <h4 className="font-medium">{movie.title}</h4>
                            <div className="text-sm text-gray-600">
                              {movie.genres.join(', ')}
                            </div>
                          </div>
                          <div>
                            <RatingStars rating={movie.predictedRating} />
                            <div className="text-xs text-gray-500 text-right mt-1">
                              Várható értékelés
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default MovieRecommender;