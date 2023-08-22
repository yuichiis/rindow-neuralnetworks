<?php
namespace Rindow\NeuralNetworks\Metric;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;

class MetricCatalog
{
    protected static $catalog = [
        'binary_accuracy' => BinaryAccuracy::class,
        'categorical_accuracy' => CategoricalAccuracy::class,
        'mean_norm2_error' => MeanNorm2Error::class,
        'mse' => MeanSquaredError::class,
        'sparse_categorical_accuracy' => SparseCategoricalAccuracy::class,
    ];

    static function factory(object $backend,string $name) : Metri
    {
        if(isset(self::$catalog[$name])) {
            $name = self::$catalog[$name];
        }
        $metric = new $name($backend);
        return $metric;
    }
}
