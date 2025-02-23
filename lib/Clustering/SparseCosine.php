<?php

declare(strict_types=1);
namespace OCA\Recognize\Clustering;

use \OCA\Recognize\Vendor\Rubix\ML\DataType;
use \OCA\Recognize\Vendor\Rubix\ML\Kernels\Distance\Distance;

/**
 * Sparse Cosine
 *
 * A version of the Cosine distance kernel that is specifically optimized for sparse vectors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino (speed optimization by Sami Finnilä)
 */
class SparseCosine implements Distance
{
    /**
     * Return the data types that this kernel is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Compute the distance between two vectors.
     *
     * @internal
     *
     * @param list<int|float> $a
     * @param list<int|float> $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $sigma = $ssA = $ssB = 0.0;

        $count = count($a);
        for ($i = 0; $i < $count; $i++) {
            $valueA = $a[$i];
            $valueB = $b[$i];

            if ($valueA != 0 and $valueB != 0) {
                $sigma += $valueA * $valueB;

                $ssA += $valueA ** 2;
                $ssB += $valueB ** 2;
            } else {
                if ($valueA != 0) {
                    $ssA += $valueA ** 2;
                }

                if ($valueB != 0) {
                    $ssB += $valueB ** 2;
                }
            }
        }

        if ($ssA === 0.0 and $ssB === 0.0) {
            return 0.0;
        }

        if ($ssA === 0.0 or $ssB === 0.0) {
            return 2.0;
        }

        return 1.0 - ($sigma / sqrt($ssA * $ssB));
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Sparse Cosine';
    }
}
